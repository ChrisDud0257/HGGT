import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import random
import numpy as np
from torch.nn import functional as F
import math
import torch.nn as nn

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop

class LocNet(torch.nn.Module):
    def __init__(self):
        super(LocNet, self).__init__()

        ch = 9**2 *3 + 7**2 *3
        self.layer1 = nn.Linear(ch, ch*2)
        self.bn1 = nn.BatchNorm1d(ch*2)
        self.layer2 = nn.Linear(ch*2, ch*2)
        self.bn2 = nn.BatchNorm1d(ch*2)
        self.layer3 = nn.Linear(ch*2, ch)
        self.bn3 = nn.BatchNorm1d(ch)
        self.layer4 = nn.Linear(ch, 6)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                # print(classname)
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.layer4(x)
        return x

@MODEL_REGISTRY.register()
class AdaTarget_Model(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(AdaTarget_Model, self).__init__(opt)
        self.queue_size = opt.get('queue_size', 144)
        if self.opt.get('Use_sharpen', None) is not None:
            self.usm_sharpener = USMSharp().cuda()

        self.pre_pad = self.opt['pre_pad']
        self.tile_size = self.opt['tile_size']
        self.tile_pad = self.opt['tile_pad']

        self.k_size = self.opt['k_size']
        self.s_size = self.opt['s_size']

        self.jpeger = DiffJPEG(differentiable=False).cuda()
        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        if opt.get('network_loc', None) is not None:
            self.net_loc = build_network(opt['network_loc'])
            self.net_loc = self.model_to_device(self.net_loc)
            self.print_network(self.net_loc)

        self.use_network_d = False

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            if self.opt['load_mode_g'] == 'original':
                net_g = torch.load(load_path)
                self.net_g.load_state_dict(net_g, strict=True)
            elif self.opt['load_mode_g'] == 'my_pretrain':
                param_key = self.opt['path'].get('param_key_g', 'params')
                self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if opt.get('network_loc', None) is not None:
            load_path_loc = self.opt['path_loc'].get('pretrain_network_loc', None)
            if load_path_loc is not None:
                if self.opt['load_mode_loc'] == 'original':
                    net_loc = torch.load(load_path_loc)
                    self.net_loc.load_state_dict(net_loc.state_dict(), strict=True)
                elif self.opt['load_mode_loc'] == 'my_pretrain':
                    param_key = self.opt['path_loc'].get('param_key_loc', 'params')
                    self.load_network(self.net_loc, load_path_loc, self.opt['path_loc'].get('strict_load_loc', True), param_key)

        if self.opt.get('network_d', None) is not None:
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)

            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_d', 'params')
                self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

            self.use_network_d = True

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.net_g.train()
        self.net_loc.train()

        if self.use_network_d:
            self.net_d.train()
            self.net_d_iters = train_opt.get('net_d_iters', 1)
            self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                if self.opt['load_mode_g'] == 'original':
                    net_g_ema = torch.load(load_path)
                    self.net_g_ema.load_state_dict(net_g_ema, strict=True)
                elif self.opt['load_mode_g'] == 'my_pretrain':
                    self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('adatarget_opt'):
            self.cri_adatarget = build_loss(train_opt['adatarget_opt']).to(self.device)
        else:
            self.cri_adatarget = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer loc
        optim_type = train_opt['optim_loc'].pop('type')
        self.optimizer_loc = self.get_optimizer(optim_type, self.net_loc.parameters(), **train_opt['optim_loc'])
        self.optimizers.append(self.optimizer_loc)

        if self.use_network_d:
            # optimizer d
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.kernel1 = data['kernel1'].to(self.device)
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        ori_h, ori_w = self.gt.size()[2:4]
        if self.opt.get('Use_sharpen', None) is not None:
            self.gt_usm = self.usm_sharpener(self.gt)

        # ----------------------- The first degradation process ----------------------- #
        # blur
        if self.opt.get('Use_sharpen', None) is not None:
            out = filter2D(self.gt_usm, self.kernel1)
        else:
            out = filter2D(self.gt, self.kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.opt['gray_noise_prob']
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)

        # clamp and round
        self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # random crop
        gt_size = self.opt['gt_size']


        self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size,
                                                             self.opt['scale'])

        # training pair pool
        self._dequeue_and_enqueue()
        if self.opt.get('Use_sharpen', None) is not None:
            self.gt_usm = self.usm_sharpener(self.gt)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    def optimize_parameters(self, current_iter):

        if self.opt.get('Use_sharpen', None) is not None and self.opt.get('l1_gt_usm', False) is True:
            self.adatarget_gt = self.gt_usm
        else:
            self.adatarget_gt = self.gt
        if self.opt.get('Use_sharpen', None) is not None and self.opt.get('percep_gt_usm', False) is True:
            self.per_gt = self.gt_usm
        else:
            self.per_gt = self.gt
        if self.opt.get('Use_sharpen', None) is not None and self.opt.get('gan_gt_usm', False) is True:
            self.gan_gt = self.gt_usm
        else:
            self.gan_gt = self.gt

        l_g_total = 0 # total loss, include L1Loss, perceptual loss, GAN loss and so on
        loss_dict = OrderedDict()

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        self.optimizer_loc.zero_grad()

        #AdaTarget
        ds = self.s_size - self.k_size

        _, _, H, W = self.adatarget_gt.size()
        unfold_H = F.unfold(F.pad(self.adatarget_gt, [ds // 2, ds // 2, ds // 2, ds // 2], mode='reflect'), self.s_size, dilation=1,
                            padding=0, stride=self.k_size)  # ([B, 363, N])
        B, _, N = unfold_H.size()
        unfold_H = unfold_H.permute(0, 2, 1).reshape(B * N, -1)

        unfold_S = F.unfold(self.output, self.k_size, dilation=1, padding=0, stride=self.k_size)  # B, C*k_size*k_size, L
        B, _, N = unfold_S.size()
        unfold_S = unfold_S.permute(0, 2, 1).reshape(B * N, -1)

        param_Loc = self.net_loc(torch.cat([unfold_S, unfold_H], 1))  # N', 6
        param_Loc = param_Loc.unsqueeze(2).view(-1, 2, 3)  # scale, theta, tx, ty

        # Generate new SR w.r.t. the current GT, instead of generating new GT.
        # See the right upper part of the page 5 of the paper.
        grid = F.affine_grid(param_Loc, torch.Size((B * N, 3, self.k_size, self.k_size)), align_corners=None)
        transformed_S = F.grid_sample(unfold_S.reshape(-1, 3, self.k_size, self.k_size), grid, padding_mode='border',
                                      align_corners=None)

        transformed_S = transformed_S.reshape(B, N, -1).permute(0, 2, 1)
        transformed_S = F.fold(transformed_S, [H, W], self.k_size, stride=self.k_size)


        if not self.use_network_d:

            l_adatarget =self.cri_adatarget(transformed_S, self.adatarget_gt)

            l_g_total += l_adatarget
            loss_dict['l_adatarget'] = l_adatarget

            l_g_total.backward()

            torch.nn.utils.clip_grad_norm_(self.net_loc.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.1)

            self.optimizer_loc.step()
            self.optimizer_g.step()

        else:
            # optimize net_g
            for p in self.net_d.parameters():
                p.requires_grad = False

            if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):

                l_adatarget = self.cri_adatarget(transformed_S, self.adatarget_gt)

                l_g_total += l_adatarget
                loss_dict['l_adatarget'] = l_adatarget

                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep, l_g_style = self.cri_perceptual(self.output, self.per_gt)
                    if l_g_percep is not None:
                        l_g_total += l_g_percep
                        loss_dict['l_g_percep'] = l_g_percep
                    if l_g_style is not None:
                        l_g_total += l_g_style
                        loss_dict['l_g_style'] = l_g_style

                # gan loss
                fake_g_pred = self.net_d(self.output)
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan

                l_g_total.backward()

                torch.nn.utils.clip_grad_norm_(self.net_loc.parameters(), 0.1)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.1)

                self.optimizer_loc.step()
                self.optimizer_g.step()

            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.gan_gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    @torch.no_grad()
    def feed_val_data(self, data):
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def pre_process(self, img):
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        self.img = img.unsqueeze(0).to(self.device)
        if self.half:
            self.img = self.img.half()

        # pre_pad
        if self.pre_pad != 0:
            self.img = F.pad(self.img, (0, self.pre_pad, 0, self.pre_pad), 'reflect')
        # mod pad
        self.scale = self.opt['scale']
        if self.scale == 2:
            self.mod_scale = 2
        elif self.scale == 1:
            self.mod_scale = 4
        if self.mod_scale is not None:
            self.mod_pad_h, self.mod_pad_w = 0, 0
            _, _, h, w = self.img.size()
            if (h % self.mod_scale != 0):
                self.mod_pad_h = (self.mod_scale - h % self.mod_scale)
            if (w % self.mod_scale != 0):
                self.mod_pad_w = (self.mod_scale - w % self.mod_scale)
            self.img = F.pad(self.img, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def tile_process(self, lq):
        """Modified from: https://github.com/ata4/esrgan-launcher
        """
        self.scale = self.opt['scale']
        self.img = lq
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.tile_size
                ofs_y = y * self.tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.tile_pad, 0)
                input_end_x_pad = min(input_end_x + self.tile_pad, width)
                input_start_y_pad = max(input_start_y - self.tile_pad, 0)
                input_end_y_pad = min(input_end_y + self.tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        if hasattr(self, 'net_g_ema'):
                            output_tile = self.net_g_ema(input_tile)
                        else:
                            output_tile = self.net_g(input_tile)
                except Exception as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return self.output

    def post_process(self):
        # remove extra pad
        if self.mod_scale is not None:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]
        # remove prepad
        if self.pre_pad != 0:
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - self.pre_pad * self.scale, 0:w - self.pre_pad * self.scale]
        return self.output

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if self.opt['tile_process']:
                    self.output = self.tile_process(self.lq)
                else:
                    self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.opt['tile_process']:
                    self.output = self.tile_process(self.lq)
                else:
                    self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            if 'lq_path' in val_data:
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            elif 'gt_path' in val_data:
                img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            self.feed_val_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_loc, 'net_loc', current_iter)
        if self.use_network_d:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
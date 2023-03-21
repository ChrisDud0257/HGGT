import os.path
import logging
import torch
import argparse

from utils import utils_logger
from utils import utils_image as util
# from utils import utils_model
from models.network_rrdbnet import RRDBNet as net


"""
Spyder (Python 3.6-3.7)
PyTorch 1.4.0-1.8.1
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/BSRGAN
        https://github.com/cszn/KAIR
If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)
by Kai Zhang ( March/2020 --> March/2021 --> )
This work was previously submitted to CVPR2021.

# --------------------------------------------
@inproceedings{zhang2021designing,
  title={Designing a Practical Degradation Model for Deep Blind Image Super-Resolution},
  author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},
  booktitle={arxiv},
  year={2021}
}
# --------------------------------------------

"""


def main(args):

    # utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    # logger = logging.getLogger('blind_sr_log')

#    print(torch.__version__)               # pytorch version
#    print(torch.version.cuda)              # cuda version
#    print(torch.backends.cudnn.version())  # cudnn version

    save_results = True
    sf = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    model_path = args.model_path          # set model path
    # logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

    # torch.cuda.set_device(0)      # set GPU ID
    # logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
    torch.cuda.empty_cache()

    # --------------------------------
    # define network and load model
    # --------------------------------
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network

#            model_old = torch.load(model_path)
#            state_dict = model.state_dict()
#            for ((key, param),(key2, param2)) in zip(model_old.items(), state_dict.items()):
#                state_dict[key2] = param
#            model.load_state_dict(state_dict, strict=True)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    torch.cuda.empty_cache()

    save_path = os.path.join(args.save_path, args.exp_name, 'visulization', f'{args.testing_set_name}')
    os.makedirs(save_path, exist_ok=True)

    # logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
    # logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
    idx = 0

    for img in util.get_image_paths(args.LQ_img_path):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d} --> {:<s} --> x{:<d}--> {:<s}'.format(idx, model_name, sf, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        # --------------------------------
        # (2) inference
        # --------------------------------
        img_E = model(img_L)

        # --------------------------------
        # (3) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)
        if save_results:
            util.imsave(img_E, os.path.join(save_path, img_name+"_"+args.suffix+".png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='././experiments/pretrained_models/BSRGAN/Blind_PSNR/RRDB_DF2K_OST_Blind_PSNR_x4.pth', help='Path to the model')
    parser.add_argument('--LQ_img_path', type=str, default='././datasets/Test/Blind_Degradation_Benchmark_LQ/Test-100/Blind_LR')
    parser.add_argument('--testing_set_name', type = str, default='Test-100')
    parser.add_argument('--save_path', type=str, default='././results')
    parser.add_argument('--exp_name', type=str, default='BSRNet_DF2K_OST_Blind_x4')
    parser.add_argument('--suffix', type=str, default='BSRNetDF2KBlind')
    args = parser.parse_args()
    main(args)

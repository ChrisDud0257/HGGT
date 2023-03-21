#################
####Train&Test AdaTarget-GAN
#1.Pre-train a model on DF2K_OST dataset with blind degradation factors and without a discriminator.
#It pre-trains a model with blind degradations without discriminator, and the model is used for Step2 and Step3 to fine-tune with GAN settings.
#This step is optional, you could just skip this step and ulitize our provided well-trained model in "experiments/pretrained_models/AdaTarget-GAN/Blind_PSNR"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train_adatarget.py -opt ./options/train/PosNegGTSR/train_AdaTarget_DF2K_OST_Blind_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_AdaTarget_DF2K_OST_Blind_x4.yml

#2.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on DF2K_OST dataset with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/AdaTarget-GAN/Blind_PSNR" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_AdaTargetGAN_DF2K_OST_Blind_x4.yml", please just modify the "path" and "path_loc"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train_adatarget.py -opt ./options/train/PosNegGTSR/train_AdaTargetGAN_DF2K_OST_Blind_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_AdaTargetGAN_DF2K_OST_Blind_x4.yml

#3.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on our proposed PosNegGT dataset(with positive GT only) with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/AdaTarget-GAN/Blind_PSNR" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_AdaTargetGAN_DF2K_OST_Blind_x4.yml", please just modify the "path" and "path_loc"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train_adatarget.py -opt ./options/train/PosNegGTSR/train_AdaTargetGAN_PosNegGT_Blind_Pos_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_AdaTargetGAN_PosNegGT_Blind_Pos_x4.yml
#################



#################
####Train&Test BSRGAN        We apply BSRGAN from the original code https://github.com/cszn/BSRGAN
#1.Pre-train a model on DF2K_OST dataset with blind degradation factors and without a discriminator.
#It pre-trains a model with blind degradations without discriminator, and the model is used for Step2 and Step3 to fine-tune with GAN settings.
#This step is optional, you could just skip this step and ulitize our provided well-trained model in "experiments/pretrained_models/BSRGAN/Blind_PSNR"
#CUDA_VISIBLE_DEVICES=0 \
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 ./train_BSGRAN/main_train_psnr.py --opt ./train_BSGRAN/options/BSRGAN/train_BSRNet_DF2K_OST_Blind_x4.json --dist True

#CUDA_VISIBLE_DEVICES=0 \
#python ./train_BSGRAN/main_test_bsrgan.py --model_path ././experiments/pretrained_models/BSRGAN/Blind_PSNR/RRDB_DF2K_OST_Blind_PSNR_x4.pth --exp_name BSRNet_DF2K_OST_Blind_x4 --suffix BSRNetDF2KBlind

#2.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on DF2K_OST dataset with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/BSRGAN/Blind_PSNR" and modify the
#parameters in the file "./train_BSGRAN/options/BSRGAN/train_BSRGAN_DF2K_OST_Blind_x4.json", please just modify the "pretrained_netG"
#CUDA_VISIBLE_DEVICES=0 \
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 ./train_BSGRAN/main_train_psnr.py --opt ./train_BSGRAN/options/BSRGAN/train_BSRGAN_DF2K_OST_Blind_x4.json --dist True

#CUDA_VISIBLE_DEVICES=0 \
#python ./train_BSGRAN/main_test_bsrgan.py --model_path ././experiments/pretrained_models/BSRGAN/Blind_GAN_DF2K_OST/RRDB_DF2K_OST_GAN_x4.pth --exp_name BSRGAN_DF2K_OST_Blind_x4 --suffix BSRGANDF2KBlind

#3.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on our proposed PosNegGT dataset(with positive GT only) with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/LDL/Blind_PSNR/RRDB_DF2K_OST_Blind_PSNR_x4.pth" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_LDL_PosNegGT_Blind_Pos_x4.yml", please just modify the "path"
#CUDA_VISIBLE_DEVICES=0 \
#python -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 ./train_BSGRAN/main_train_psnr.py --opt ./train_BSGRAN/options/BSRGAN/train_BSRGAN_PosNegGT_Blind_Pos_x4.json --dist True

#CUDA_VISIBLE_DEVICES=0 \
#python ./train_BSGRAN/main_test_bsrgan.py --model_path ././experiments/pretrained_models/BSRGAN/Blind_GAN_PosNegGT_Pos/RRDB_PosNegGT_Pos_GAN_x4.pth --exp_name BSRGAN_PosNegGT_Blind_Pos_x4 --suffix BSRGANPosNegGTBlindPos
#################



#################
####Train&Test LDL
#1.Pre-train a model on DF2K_OST dataset with blind degradation factors and without a discriminator.
#It pre-trains a model with blind degradations without discriminator, and the model is used for Step2 and Step3 to fine-tune with GAN settings.
#This step is optional, you could just skip this step and ulitize our provided well-trained model in "experiments/pretrained_models/AdaTarget-GAN/Blind_PSNR/RRDB_DF2K_OST_Blind_PSNR_x4.pth" in Step2 and Step3
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_LDL_noGAN_DF2K_OST_Blind_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_LDL_noGAN_DF2K_OST_Blind_x4.yml

#2.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on DF2K_OST dataset with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/LDL/Blind_PSNR" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_LDL_DF2K_OST_Blind_x4.yml", please just modify the "path"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_LDL_DF2K_OST_Blind_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_LDL_DF2K_OST_Blind_x4.yml

#3.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on our proposed PosNegGT dataset(with positive GT only) with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/LDL/Blind_PSNR/RRDB_DF2K_OST_Blind_PSNR_x4.pth" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_LDL_PosNegGT_Blind_Pos_x4.yml", please just modify the "path"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_LDL_PosNegGT_Blind_Pos_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_LDL_PosNegGT_Blind_Pos_x4.yml
#################


#################
####Train&Test Real-ESRGAN
#1.Pre-train a model on DF2K_OST dataset with blind degradation factors and without a discriminator.
#It pre-trains a model with blind degradations without discriminator, and the model is used for Step2 and Step3 to fine-tune with GAN settings.
#This step is optional, you could just skip this step and ulitize our provided well-trained model in "experiments/pretrained_models/Real-ESRGAN/Blind_PSNR" in Step2 and Step3
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_RealESRNet_DF2K_OST_Blind_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_RealESRNet_DF2K_OST_Blind_x4.yml

#2.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on DF2K_OST dataset with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/Real-ESRGAN/Blind_PSNR" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_RealESRGAN_DF2K_OST_Blind_x4.yml", please just modify the "path"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_RealESRGAN_DF2K_OST_Blind_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_RealESRGAN_DF2K_OST_Blind_x4.yml

#3.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on our proposed PosNegGT dataset(with positive GT only) with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/Real-ESRGAN/Blind_PSNR/RRDB_DF2K_OST_Blind_PSNR_x4.pth" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_RealESRGAN_DF2K_OST_Blind_x4.yml", please just modify the "path"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_RealESRGAN_PosNegGT_Blind_Pos_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_RealESRGAN_PosNegGT_Blind_Pos_x4.yml
#################


#################
####Train&Test RRDB-GAN
#1.Pre-train a model on DF2K_OST dataset with blind degradation factors and without a discriminator.
#It pre-trains a model with blind degradations without discriminator, and the model is used for Step2 and Step3 to fine-tune with GAN settings.
#This step is optional, you could just skip this step and ulitize our provided well-trained model in "experiments/pretrained_models/RRDB-GAN/Blind_PSNR/RRDB_DF2K_OST_Blind_PSNR_x4.pth" in Step2 and Step3
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_RRDB_DF2K_OST_Blind_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_RRDB_DF2K_OST_Blind_x4.yml

#2.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on our proposed dataset(with original GT only) with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/RRDB-GAN/Blind_PSNR/RRDB_DF2K_OST_Blind_PSNR_x4.pth" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Ori_x4.yml", please just modify the "path"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Ori_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_RRDBGAN_PosNegGT_Blind_Ori_x4.yml

#3.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on our proposed PosNegGT dataset(with positive GT only) with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/RRDB-GAN/Blind_PSNR/RRDB_DF2K_OST_Blind_PSNR_x4.pth" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Pos_x4.yml", please just modify the "path"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Pos_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_RRDBGAN_PosNegGT_Blind_Pos_x4.yml

#4.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on our proposed PosNegGT dataset(with both positive and negative GT) with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/RRDB-GAN/Blind_PSNR/RRDB_DF2K_OST_Blind_PSNR_x4.pth" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Pos+Neg_x4.yml", please just modify the "path"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Pos+Neg_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_RRDBGAN_PosNegGT_Blind_Pos+Neg_x4.yml
#################


#################
####Train&Test SwinIR-GAN
#1.Pre-train a model on DF2K_OST dataset with blind degradation factors and without a discriminator.
#It pre-trains a model with blind degradations without discriminator, and the model is used for Step2 and Step3 to fine-tune with GAN settings.
#This step is optional, you could just skip this step and ulitize our provided well-trained model in "experiments/pretrained_models/SwinIR-GAN/Blind_PSNR/SwinIR_DF2K_OST_Blind_PSNR_x4.pth" in Step2 and Step3
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_SwinIR_DF2K_OST_Blind_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_SwinIR_DF2K_OST_Blind_x4.yml

#2.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on our proposed dataset(with original GT only) with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/SwinIR-GAN/Blind_PSNR/SwinIR_DF2K_OST_Blind_PSNR_x4.pth" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_SwinIRGAN_PosNegGT_Blind_Ori_x4.yml", please just modify the "path"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_SwinIRGAN_PosNegGT_Blind_Ori_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_SwinIRGAN_PosNegGT_Blind_Ori_x4.yml

#3.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on our proposed PosNegGT dataset(with positive GT only) with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/SwinIR-GAN/Blind_PSNR/SwinIR_DF2K_OST_Blind_PSNR_x4.pth" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_SwinIRGAN_PosNegGT_Blind_Pos_x4.yml", please just modify the "path"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_SwinIRGAN_PosNegGT_Blind_Pos_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_SwinIRGAN_PosNegGT_Blind_Pos_x4.yml

#4.Fine-tune a GAN-based model with the PSNR-model pre-trained in Step1 on our proposed PosNegGT dataset(with both positive and negative GT) with blind degradation factors together with a discriminator.
#If you skip Step1, you could just utilize our provided well-trained model in "experiments/pretrained_models/SwinIR-GAN/Blind_PSNR/SwinIR_DF2K_OST_Blind_PSNR_x4.pth" and modify the
#parameters in the file "./options/train/PosNegGTSR/train_SwinIRGAN_PosNegGT_Blind_Pos+Neg_x4.yml", please just modify the "path"
#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_SwinIRGAN_PosNegGT_Blind_Pos+Neg_x4.yml --auto_resume

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_SwinIRGAN_PosNegGT_Blind_Pos+Neg_x4.yml
#################



#################
####Evaluation with metrics
#We build up a Test-100 testding dataset, which have 100 sets of images, every set has one original GT images together with their 4 enhanced versions.
#In each set, the original GT image has at least two postive enhanced versions.
#For each SR result, we compute the metrics with the corresponding enhanced positive GT versions and then average all of the values which are computed with the different positive GTs.
#You only need to change the restored path, which means you save your SR results.
#PSNR
#python ./scripts/metrics/calculate_multigt_labeled_psnr_ssim.py --restored ../../results/BSRGAN_DF2K_OST_Blind_x4/visulization/Test-100

#LPIPS
#python ./scripts/metrics/calculate_multigt_labeled_lpips.py --restored ../../results/BSRGAN_DF2K_OST_Blind_x4/visulization/Test-100

#DISTS
#python ./scripts/metrics/calculate_multigt_labeled_dists.py --restored ../../results/BSRGAN_DF2K_OST_Blind_x4/visulization/Test-100
#################
####Step 1: Before training, firstly you should generate Trainging GT dataset meta information
###########################
#python ./scripts/data_preparation/generate_meta_info.py
###########################




####Step 2: Train Enhancement Moodels
###########################
##For a single GPU

#CUDA_VISIBLE_DEVICES=0 \
#python ./basicsr/train.py -opt ./options/train/EnhancementModels/train_EnhanceRCAN_01.yml --auto_resume

#CUDA_VISIBLE_DEVICES=1 \
#python ./basicsr/train.py -opt ./options/train/EnhancementModels/train_EnhanceRCAN_02.yml --auto_resume

#CUDA_VISIBLE_DEVICES=2 \
#python ./basicsr/train.py -opt ./options/train/EnhancementModels/train_EnhanceELAN_03.yml --auto_resume

#CUDA_VISIBLE_DEVICES=3 \
#python ./basicsr/train.py -opt ./options/train/EnhancementModels/train_EnhanceELAN_04.yml --auto_resume

##For Multiple GPUs

#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 ./basicsr/train.py -opt ./options/train/EnhancementModels/train_EnhanceRCAN_01.yml --launcher pytorch

#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 ./basicsr/train.py -opt ./options/train/EnhancementModels/train_EnhanceRCAN_02.yml --launcher pytorch

#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 ./basicsr/train.py -opt ./options/train/EnhancementModels/train_EnhanceELAN_03.yml --launcher pytorch

#CUDA_VISIBLE_DEVICES=0,1,2,3 \
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 ./basicsr/train.py -opt ./options/train/EnhancementModels/train_EnhanceELAN_04.yml --launcher pytorch
###########################




####Step 3: Inferencing enhanced GTs
###########################
#CUDA_VISIBLE_DEVICES=0 \
#python ./inference/inference_rcannoup.py --input /home/chendu/data2_hdd10t/chendu/dataset/ForMultiGT/2/MultiGT1600 \
#--output ./results/EnhancedGT/01 --model_path ./experiments/pretrained_models/enhancement/RCAN_01.pth --tile_size 1200 --suffix _01

#CUDA_VISIBLE_DEVICES=1 \
#python ./inference/inference_rcannoup.py --input /home/chendu/data2_hdd10t/chendu/dataset/ForMultiGT/2/MultiGT1600 \
#--output ./results/EnhancedGT/02 --model_path ./experiments/pretrained_models/enhancement/RCAN_02.pth --tile_size 1200 --suffix _02

#CUDA_VISIBLE_DEVICES=2 \
#python ./inference/inference_elan.py --input /home/chendu/data2_hdd10t/chendu/dataset/ForMultiGT/2/MultiGT1600 \
#--output ./results/EnhancedGT/03 --model_path ./experiments/pretrained_models/enhancement/ELAN_03.pth --tile_size 1200 --suffix _03

#CUDA_VISIBLE_DEVICES=3 \
#python ./inference/inference_elan.py --input /home/chendu/data2_hdd10t/chendu/dataset/ForMultiGT/2/MultiGT1600 \
#--output ./results/EnhancedGT/04 --model_path ./experiments/pretrained_models/enhancement/ELAN_04.pth --tile_size 1200 --suffix _04
###########################

# HGGT

Official PyTorch code and dataset for our paper "HGGT" in CVPR 2023.

### [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Human_Guided_Ground-Truth_Generation_for_Realistic_Image_Super-Resolution_CVPR_2023_paper.pdf) | [Supplementary](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Chen_Human_Guided_Ground-Truth_CVPR_2023_supplemental.pdf) | [Arxiv Version](https://arxiv.org/abs/2303.13069)

> **Human Guided Ground-truth Generation for Realistic Image Super-resolution** <br>
> [Du CHEN\*](https://github.com/ChrisDud0257), [Jie LIANG\*](https://liangjie.xyz/), Xindong ZHANG, Ming LIU, Hui ZENG and [Lei ZHANG](https://www4.comp.polyu.edu.hk/~cslzhang/). <br>
> Accepted by CVPR 2023.<br>


## Abstract
How to generate the ground-truth (GT) image is a critical
issue for training realistic image super-resolution (Real-
ISR) models. Existing methods mostly take a set of highresolution
(HR) images as GTs and apply various degradations
to simulate the low-resolution (LR) counterparts.
Though great progress has been achieved, such an LR-HR
pair generation scheme has several limitations. First, the
perceptual quality of HR images may not be high enough,
limiting the quality of Real-ISR model output. Second, existing
schemes do not consider much human perception in GT
generation, and the trained models tend to produce oversmoothed
results or unpleasant artifacts. With the above
considerations, we propose a human guided GT generation
scheme. We first elaborately train multiple image enhancement
models to improve the perceptual quality of HR images,
and enable one LR image having multiple HR counterparts.
Human subjects are then involved to annotate
the high quality regions among the enhanced HR images
as GTs, and label the regions with unpleasant artifacts as
negative samples. A human guided GT image dataset with
both positive and negative samples is then constructed, and
a loss function is proposed to train the Real-ISR models.
Experiments show that the Real-ISR models trained on our
dataset can produce perceptually more realistic results with
less artifacts.

## Overall illustration of our dataset generation:
![sketch map](./figures/sketch%20map.png)

For more details, please refer to our paper.

## Getting started
- Clone this repo.
```bash
git clone https://github.com/ChrisDud0257/HGGT
cd HGGT
```

- We recommend that you just skip to Step 4 if you don't have enough time or energy to go through the whole dataset generation procedures from Step 1 to Step 3. So you could directly utilize our proposed dataset to train your own models!
- If you are interested in our data generation steps, then you are welcome to follow our integral dataset generation progress. 
- We will also provide our data annotation software as well as the annotation tutorial in the future. We hope our attempts and experiences will help more researchers to generate their human guided dataset and extand our ideas to their research fields so as to make great success in the GT manipulation.
- All of the datasets, models, results could be download through [GoogleDrive](https://drive.google.com/drive/folders/1KwgbmWhWuAqvIf1Yzi5a2P-xQYX9PuO_?usp=sharing).

### Step 1: Image enhancement (*Optional*, you could just skip to step4 and train models with our proposed HGGT dataset)
#### Installation. (Python 3 + NVIDIA GPU + CUDA. Recommend to use Anaconda)
```bash
cd ImageEnhancement
conda create --name enhance python=3.9
pip install -r requirements.txt
python setup.py develop
```
We build up our project based on the widely-used [BasicSR](https://github.com/XPixelGroup/BasicSR) codeframe. For more installation details, please refer to the [BasicSR](https://github.com/XPixelGroup/BasicSR) framework.
- Prepare the training and testing dataset by following this [instruction](ImageEnhancement/datasets/README.md).
- Prepare the pre-trained models by following this [instruction](ImageEnhancement/experiments/README.md).

#### Training
- Firstly, check and modify the yml file ```./ImageEnhancement/options/train/EnhancementModels/train_EnhanceRCAN_01.yml```, this train options folder has four different enhancement models, you could just modify the parameters according to your own requirements.
- Secondly, uncomment the commands in ```./ImageEnhancement/demo.sh```. For example:
```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/EnhancementModels/train_EnhanceRCAN_01.yml --auto_resume
```
- And then,
```bash
cd ImageEnhancement
sh demo.sh
```


#### Inference

- Firstly, please follow this [instruction](./ImageEnhancement/datasets/README.md) to prepare the dataset for inference.
- (*Optional*) Secondly, please follow this [instruction](./ImageEnhancement/experiments/README.md) to prepare our well-trained models for inference. Or you could just utilize your own trained models.
- Uncomment the commands in ```./ImageEnhancement/demo.sh``` and modify ```--input```, ```--output``` and ```--model_path```. For example:
```bash
CUDA_VISIBLE_DEVICES=0 \
python ./inference/inference_rcannoup.py --input /home/chendu/data2_hdd10t/chendu/dataset/ForMultiGT/2/MultiGT1600 \
--output ./results/EnhancedGT/01 --model_path ./experiments/pretrained_models/enhancement/RCAN_01.pth --tile_size 1200 --suffix _01
```
- And then,
```bash
cd ImageEnhancement
sh demo.sh
```
- If your GPU memory is limited, please reduce ```--tile_size``` to ```800``` or even smaller.
- We provide the full size enhanced GT images which are prapared for the patch selection in Step 2. For directly usage, please follow this [instruction](./ImageEnhancement/datasets/README.md).
- **Note that, if you want to try our data annotation software, please make sure the four enhanced images generated by four different enhance models are named ended with "_01", "_02", "_03", "_04" and be put into the corresponding folder "01", "02", "03", "04". The original GT images should also be put into the "original" folder.** For example, the folder structure should be the following:
```
-EnhancedGT/
    -original/
        -img1.png
    -01/
        -img1_01.png
    -02/
        -img1_02.png
    -03/
        -img1_03.png
    -04/
        -img1_04.png
```
- "Original" folder means it has all of the original GT images. If the enhanced and original images are not placed in this structure, then the annotation software will not work.


### Step 2: Patch selection (*Optional*)
#### We complete Step 2 and Step 3 on Windows Operating System.

- Firstly, please put all of the enhanced GTs and original GTs into the folder structure as has been illustrated in Step 1.
- Secondly, complete the patch selection through our code. Please modify ```--img_path``` and ```--save_path```. 
- We select patches according to the quantity of details and textures, which is measured by the standard deviation (std) of the patch in image domain and the std of high-frequency component in a Laplacian pyramid, accordingly filter out the patches which have large smooth background regions.Thus, ```--cont_var_thresh``` and ```--freq_var_thresh``` control the final total saved amounts of the patches, when they become large, the patches with little edge information will be filtered out, and less patches would be saved.
```bash
cd PatchSelection
python patch_select.py --img_path 'Path to EnhancedGT' --save_path 'Path to your own patch selection save path'
```
- The save path folder structures will be as follows:
```
-PatchSelection/
    -original/
        -img1-x-y.png
        -img1-x1-y1.png
    -01/
        -img1-x-y_01.png
        -img1-x1-y1_01.png
    -02/
        -img1-x-y_02.png
        -img1-x1-y1_02.png
    -03/
        -img1-x-y_03.png
        -img1-x1-y1_03.png
    -04/
        -img1-x-y_04.png
        -img1-x1-y1_04.png
```
- Where ```x, y, x1, y1``` record the top-left pixel position of the selected patches.
- Then you are recommended to filter out the patches manually for which the difference between the original version and the enhanced version is small (i.e., no much enhancement).
- In order to annotate the GTs more efficiently in Step 3, you are recommanded to disorder the GTs and re-group them into multiple groups. In our annotation progress, we distribute 1000 sets of images to every volunteers. In every set, the  five GT images(i.e.original GT and the corresponding 01, 02, 03, 04 enhanced GTs) are randomly chosed without repetition from the original "PatchSelection" folder.

```bash
cd PatchSelection
python regroup.py --img_path 'Path to the patch selection save path' --save_path 'Path to your own regroup save path'
```
### Step 3: Annotation (*Optional*)
**We will provide the annotation software as well as the software tutorial in the future.**


## Ours HGGT dataset Structure

After the annotation step, we integrate the images and labels into the following folder structure:
```bash
Train/
    images/
        0-20193/
            original/
                img0-x-y.png
                img1-x1-y1.png
            01/
                img0-x-y_01.png
                img1-x1-y1_01.png
            02/
                img0-x-y_02.png
                img1-x1-y1_02.png
            03/
                img0-x-y_03.png
                img1-x1-y1_03.png
            04/
                img0-x-y_04.png
                img1-x1-y1_04.png
    labels/
        0-20193/
            A/
                img0-x-y.json
                img1-x1-y1.json
            B/
                img0-x-y.json
                img1-x1-y1.json
            C/
                img0-x-y.json
                img1-x1-y1.json
```
- For example, for each set of GTs, the original GT image ```img0-x-y.png``` has its corresponding enhanced versions, ```img0-x-y_01.png```, ```img0-x-y_02.png```, ```img0-x-y_03.png```, ```img0-x-y_04.png```, the five images are placed in the folder ```original/```, ```01/```, ```02/```, ```03/```, ```04/``` respectively.

- As for the label, in the ```labels/0-20193/``` folder, the three different folder ```A/```, ```B/```, ```C/``` means each set of images is annotated by three differnt people, we denote ```A, B, C``` for representing the three people. For example, the image set ```img0-x-y.png, img0-x-y_01.png, img0-x-y_02.png, img0-x-y_03.png, img0-x-y_04.png``` are annotated by person 'A', and the annotation information is saved in the file ```labels/0-20193/A/img0-x-y.json```. The ```img0-x-y.json``` has the same name with the original GT image ```img0-x-y.png```, and it records the total four enhanced GTs' annotation results.

- In each annotation label, the ```.json``` file has the following data structures:
```bash
Picture_1 {'Name': '0001-586-1469.png', 'File': 'original', 'Label': None}
Picture_2 {'Name': '0001-586-1469_01.png', 'File': '01', 'Label': 'Positive'}
Picture_3 {'Name': '0001-586-1469_02.png', 'File': '02', 'Label': 'Positive'}
Picture_4 {'Name': '0001-586-1469_03.png', 'File': '03', 'Label': 'Positive'}
Picture_5 {'Name': '0001-586-1469_04.png', 'File': '04', 'Label': 'Positive'}
Time_cost {'Start': '2022-08-05 16:56:06', 'End': '2022-08-05 16:56:49', 'Total': 42}
```

- For more information, such as downloading and usage for train SR models in Step 4, please refer to the [dataset instruction](PosNegGTSR/datasets/README.md).


### Step 4: Train realistic Super-resolution models on our proposed HGGT dataset
#### Installation. (Python 3 + NVIDIA GPU + CUDA. Recommend to use Anaconda)
```bash
cd PosNegGTSR
conda create --name posnegsr python=3.9
pip install -r requirements.txt
python setup.py develop
```

We build up our project based on the widely-used [BasicSR](https://github.com/XPixelGroup/BasicSR) codeframe. For more installation details, please refer to the [BasicSR](https://github.com/XPixelGroup/BasicSR) framework.
- Prepare the training and testing dataset by following this [instruction](PosNegGTSR/datasets/README.md).
- Prepare the pre-trained models by following this [instruction](PosNegGTSR/experiments/README.md).

#### Pre-train with out GAN (*Optional*)
- Following [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), we pre-train a PSNR-model on DF2K_OST dataset with blind degradation factors and without a discriminator.
- This step is optional, you could just skip this step and ulitize our provided well-trained model to fine-tune your GAN-based models with our proposed PosNegGT dataset.
- Firstly, check and modify the yml file ```./PosNegGTSR/options/train/PosNegGTSR/train_RRDB_DF2K_OST_Blind_x4.yml```.
- Secondly, uncomment the commands in ```./PosNegGTSR/demo.sh```. For example:
```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_RRDB_DF2K_OST_Blind_x4.yml --auto_resume
```
- And then,
```bash
cd PosNegGTSR
sh demo.sh
```

#### Training with a GAN with the original GT in our HGGT dataset or with DF2K_OST dataset.

- **Original GT in HGGT dataset**

- Firstly, check and modify the yml file ```./PosNegGTSR/options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Ori_x4.yml```. We fine-tune our GAN-model with a well-trained PSNR-model.
- Secondly, uncomment the commands in ```./PosNegGTSR/demo.sh```. For example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Ori_x4.yml --auto_resume
```

- And then,
```bash
cd PosNegGTSR
sh demo.sh
```

- **DF2K_OST dataset**

- Firstly, check and modify the yml file ```./PosNegGTSR/options/train/PosNegGTSR/train_LDL_DF2K_OST_Blind_x4.yml```. We fine-tune our GAN-model with a well-trained PSNR-model.
- Secondly, uncomment the commands in ```./PosNegGTSR/demo.sh```. For example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_LDL_DF2K_OST_Blind_x4.yml --auto_resume
```

- And then,
```bash
cd PosNegGTSR
sh demo.sh
```

#### Trainging with the positive GTs in our HGGT dataset.
- Firstly, check and modify the yml file ```./PosNegGTSR/options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Pos_x4.yml```. We fine-tune our GAN-model with a well-trained PSNR-model.
- Secondly, uncomment the commands in ```./PosNegGTSR/demo.sh```. For example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Pos_x4.yml --auto_resume
```

- And then,
```bash
cd PosNegGTSR
sh demo.sh
```

#### Trainging with both positive and negative GTs in our HGGT dataset.
- Firstly, check and modify the yml file ```./PosNegGTSR/options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Pos+Neg_x4.yml```. We fine-tune our GAN-model with a well-trained PSNR-model.
- Secondly, uncomment the commands in ```./PosNegGTSR/demo.sh```. For example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/train.py -opt ./options/train/PosNegGTSR/train_RRDBGAN_PosNegGT_Blind_Pos+Neg_x4.yml --auto_resume
```

- And then,
```bash
cd PosNegGTSR
sh demo.sh
```

### Step 5: Testing with the models trained in step 4

- Prepare the testing dataset by following this [instruction](PosNegGTSR/datasets/README.md).
- Prepare the pre-trained models by following this [instruction](PosNegGTSR/experiments/README.md).

To download our pre-trained models and for usage, please follow this [instruction](./PosNegGTSR/experiments/README.md).

#### Testing with the model trained on the original GT in our proposed HGGT dataset or DF2K_OST dataset.

- **Model trained with Original GT in HGGT dataset**

- Firstly, check and modify the yml file ```./PosNegGTSR/options/test/PosNegGTSR/test_RRDBGAN_PosNegGT_Blind_Ori_x4.yml```.
- Secondly, uncomment the commands in ```./PosNegGTSR/demo.sh```. For example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_RRDBGAN_PosNegGT_Blind_Ori_x4.yml
```

- And then,
```bash
cd PosNegGTSR
sh demo.sh
```

- **Model trained with DF2K_OST dataset**

- Firstly, check and modify the yml file ```./PosNegGTSR/options/test/PosNegGTSR/test_LDL_DF2K_OST_Blind_x4.yml```.
- Secondly, uncomment the commands in ```./PosNegGTSR/demo.sh```. For example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_LDL_DF2K_OST_Blind_x4.yml
```

- And then,
```bash
cd PosNegGTSR
sh demo.sh
```

- **Model trained with positive GT in our HGGT datset**

- Firstly, check and modify the yml file ```./PosNegGTSR/options/test/PosNegGTSR/test_RRDBGAN_PosNegGT_Blind_Pos_x4.yml```.
- Secondly, uncomment the commands in ```./PosNegGTSR/demo.sh```. For example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_RRDBGAN_PosNegGT_Blind_Pos_x4.yml
```

- And then,
```bash
cd PosNegGTSR
sh demo.sh
```

- **Model trained with both positive and negative GT in our HGGT datset**

- Firstly, check and modify the yml file ```./PosNegGTSR/options/test/PosNegGTSR/test_RRDBGAN_PosNegGT_Blind_Pos+Neg_x4.yml```.
- Secondly, uncomment the commands in ```./PosNegGTSR/demo.sh```. For example:

```bash
CUDA_VISIBLE_DEVICES=0 \
python ./basicsr/test.py -opt ./options/test/PosNegGTSR/test_RRDBGAN_PosNegGT_Blind_Pos+Neg_x4.yml
```

- And then,
```bash
cd PosNegGTSR
sh demo.sh
```

### Step 6: Evaluation metrics
- Prepare testing dataset by following this [instruction](PosNegGTSR/datasets/README.md).
- Our Test-100 testing dataset has 100 512*512 HR images together with their enhanced GTs. In each set of images, we place the original HR, the enhanced 01, 02, 03, 04 GTs. Every original GT has at least 2 positive enhanced GTs. 
- For each SR result, we compute the metrics with the corresponding enhanced positive GT versions and then average all of the values which are computed with the different positive GTs.

#### PSNR & SSIM & LPIPS & DISTS
- Firstly, check and modify the ```.py``` file ```./PosNegGTSR/scripts/metrics/calculate_multigt_labeled_psnr_ssim.py```,  ```./PosNegGTSR/scripts/metrics/calculate_multigt_labeled_lpips.py```, ```./PosNegGTSR/scripts/metrics/calculate_multigt_labeled_dists.py```
- Please modify ```--gts [Path to the Test-100 GT images file]```, ```--restored [path to your SR results]```, ```--json_path [Path to Test-100 label file]```. 
- Secondly, uncomment the commands in ```./PosNegGTSR/demo.sh```. For example:
```bash
python ./scripts/metrics/calculate_multigt_labeled_psnr_ssim.py
python ./scripts/metrics/calculate_multigt_labeled_lpips.py
python ./scripts/metrics/calculate_multigt_labeled_dists.py
```
- And then,
```bash
cd PosNegGTSR
sh demo.sh
```

### SR results
We also provide all of the visualization and quantitative metrics results reported in our paper, you could download through [GoogleDrive](https://drive.google.com/drive/folders/1KgA_TLdrgN3DonI_zWD06X4kFO38D9k0?usp=sharing).

## License
This project is released under the Apache 2.0 license.

## Citation

```bash
@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Du and Liang, Jie and Zhang, Xindong and Liu, Ming and Zeng, Hui and Zhang, Lei},
    title     = {Human Guided Ground-Truth Generation for Realistic Image Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14082-14091}
}
```

## Acknowledgement

This project is built mainly based on the excellent [BasicSR](https://github.com/XPixelGroup/BasicSR) and [KAIR](https://github.com/cszn/KAIR) codeframe. We appreciate it a lot for their developers.

## Contact

If you have any questions or suggestions about this project, please contact me at ```csdud.chen@connect.polyu.hk``` .
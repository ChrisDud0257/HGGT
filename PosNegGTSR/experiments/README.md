# Pretrained Models
We provide our well-trained models reported in our paper. You could utilize them to generate the SR results as in our paper or just to reconstruct your own low-quality images.
- Firstly, please download the pre-trained models from [GoogleDrive](https://drive.google.com/drive/folders/1FRB0RJSovECdJZMOVbhfr4rbLbZRNRpA?usp=sharing).
- Secondly, put the pre-trained models into ```./PosNegGTSR/experiments/```.
- During testing stage, please modify ```path[pretrain_network_g]``` to your download pretrained model path in the ```.yml``` file.
For example, in ```./PosNegGTSR/options/test/PosNegGTSR/test_RRDBGAN_PosNegGT_Blind_Pos_x4.yml```,
you should modify as follows:
```bash
path:
  pretrain_network_g: experiments/pretrained_models/RRDB-GAN/Blind_GAN_PosNegGT_Pos/RRDB_PosNegGT_Pos_GAN_x4.pth #/Path to/pretrained model 'RRDB-GAN/Blind_GAN_PosNegGT_Pos/RRDB_PosNegGT_Pos_GAN_x4.pth'
  strict_load_g: true
  param_key_g: params_ema
```
# Pre-trained Models
We provide our well-trained four enhancement models. You could utilize them to directly generate any enhanced GTs with your own HR images or our [MultiGT1600](../datasets/README.md) inferencing dataset.
- Firstly, please download the pre-trained models from [GoogleDrive](https://drive.google.com/drive/folders/1o9ZgSsv6ZkA6SB6nC0mOqynKGUygi552?usp=sharing).
- Secondly, put the pre-trained models into ```./ImageEnhancement/experiments/```.

During inference stage, please modify ```--model_path``` to your download pretrained model path in ```./ImageEnhancement/demo.sh``` or ```./ImageEnhancement/inference/inference_*.py```.
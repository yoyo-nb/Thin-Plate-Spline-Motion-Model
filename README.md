# [CVPR2022] Thin-Plate Spline Motion Model for Image Animation

Source code of the CVPR'2022 paper "[Thin-Plate Spline Motion Model for Image Animation](https://arxiv.org/abs/2203.14367)". 

### Example animation

![vox](assets/vox.gif)
![ted](assets/ted.gif)

**PS**: The paper trains the model for 100 epochs for a fair comparison. You can use more data and train for more epochs to get better performance.

### Installation

We support ```python3```.(Recommended version is Python 3.9).
To install the dependencies run:
```bash
pip install -r requirements.txt
```


### YAML configs
 
There are several configuration files one for each `dataset` in the `config` folder named as ```config/dataset_name.yaml```. See ```config/dataset.yaml``` to get the description of each parameter.

See description of the parameters in the ```config/taichi-256.yaml```.

### Datasets

1) **MGif**. Follow [Monkey-Net](https://github.com/AliaksandrSiarohin/monkey-net).

2) **TaiChiHD** and **VoxCeleb**. Follow instructions from [video-preprocessing](https://github.com/AliaksandrSiarohin/video-preprocessing). 

3) **TED-talks**. Follow instructions from [MRAA](https://github.com/snap-research/articulated-animation).


### Training
To train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0,1 python run.py --config config/dataset_name.yaml --device_ids 0,1
```
A log folder named after the timestamp will be created. Checkpoints, loss values, reconstruction results will be saved to this folder.


#### Training AVD network
To train a model on specific dataset run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --mode train_avd --checkpoint '{checkpoint_folder}/checkpoint.pth.tar' --config config/dataset_name.yaml
```
Checkpoints, loss values, reconstruction results will be saved to `{checkpoint_folder}`.



### Evaluation on video reconstruction

To evaluate the reconstruction performance run:
```
CUDA_VISIBLE_DEVICES=0 python run.py --mode reconstruction --config config/dataset_name.yaml --checkpoint '{checkpoint_folder}/checkpoint.pth.tar'
```
The `reconstruction` subfolder will be created in `{checkpoint_folder}`.
The generated video will be stored to this folder, also generated videos will be stored in ```png``` subfolder in loss-less '.png' format for evaluation.
To compute metrics, follow instructions from [pose-evaluation](https://github.com/AliaksandrSiarohin/pose-evaluation).



### Pre-trained models
- [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/30ab8765da364fefa101/)
- [Google Drive](https://drive.google.com/drive/folders/1pNDo1ODQIb5HVObRtCmubqJikmR7VVLT?usp=sharing)

### Image animation demo
- Google Colab: [here](https://colab.research.google.com/drive/1DREfdpnaBhqISg0fuQlAAIwyGVn1loH_?usp=sharing)
- notebook: `demo.ipynb`, edit the config cell and run for image animation.
- python:
```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --config config/vox-256.yaml --checkpoint checkpoints/vox.pth.tar --source_image ./source.jpg --driving_video ./driving.mp4
```

# Acknowledgments
The main code is based upon [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) and [MRAA](https://github.com/snap-research/articulated-animation)

Thanks for the excellent works!
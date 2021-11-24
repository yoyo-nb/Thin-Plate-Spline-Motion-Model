# Thin-Plate Spline Motion Model for Image Animation

Part of the source code of the paper "Thin-Plate Spline Motion Model for Image Animation". The full source code and pre-trained models will be publicly available.

### Installation

We support ```python3```. To install the dependencies run:
```bash
pip install -r requirements.txt
```


### YAML configs

There are several configuration files one for each `dataset` in the `config` folder named as ```config/dataset_name.yaml```. See ```config/dataset.yaml``` to get the description of each parameter.

See description of the parameters in the ```config/taichi-256.yaml```.

### Datasets

1) **MGif**. Follow [Monkey-Net](https://github.com/AliaksandrSiarohin/monkey-net).

2) **TaiChiHD** and **VoxCeleb**. Follow instructions from https://github.com/AliaksandrSiarohin/video-preprocessing. 

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
To compute metrics, follow instructions from https://github.com/AliaksandrSiarohin/pose-evaluation.



### Pre-trained models
Coming soon

### Image animation demo
Coming soon


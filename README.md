## CANet: Cross-disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading

Pytorch implementation of CANet: Cross-disease attention network. <br/>

## Paper
[CANet: Cross-disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading](https://arxiv.org/abs/1911.01376)
<br/>
IEEE Transactions on Medical Imaging
<br/>
<br/>
<p align="center">
  <img src="figure/framework.png">
</p>

## Installation
* Install Pytorch 1.1.0 and CUDA 9.0
* Clone this repo
```
git clone https://github.com/xmengli999/CANet
cd CANet
```

## Data Preparation
* Raw data needs to be written into `tfrecord` format to be decoded by `./data_loader.py`. The pre-processed data has been released from [PnP-AdaNet](https://github.com/carrenD/Medical-Cross-Modality-Domain-Adaptation). The training data can be downloaded [here](https://drive.google.com/file/d/1m9NSHirHx30S8jvN0kB-vkd7LL0oWCq3/view). The testing CT data can be downloaded [here](https://drive.google.com/file/d/1SJM3RluT0wbR9ud_kZtZvCY0dR9tGq5V/view). The testing MR data can be downloaded [here](https://drive.google.com/file/d/1RNb-4iYWUaFBY61rFAnT2XT0mtwlnH1V/view).
* Put `tfrecord` data of two domains into corresponding folders under `./data` accordingly.
* Run `./create_datalist.py` to generate the datalists containing the path of each data.

## Train
* Modify paramter values in `./config_param.json`
* Run `./main.py` to start the training process

## Evaluate
* Specify the model path and test file path in `./evaluate.py`
* Run `./evaluate.py` to start the evaluation.

## Citation
If you find the code useful for your research, please cite our paper.
```
@article{chen2020unsupervised,
  title     = {Unsupervised Bidirectional Cross-Modality Adaptation via 
               Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation},
  author    = {Chen, Cheng and Dou, Qi and Chen, Hao and Qin, Jing and Heng, Pheng Ann},
  journal   = {arXiv preprint arXiv:2002.02255},
  year      = {2020}
}

@inproceedings{chen2019synergistic,
  author    = {Chen, Cheng and Dou, Qi and Chen, Hao and Qin, Jing and Heng, Pheng-Ann},
  title     = {Synergistic Image and Feature Adaptation: 
               Towards Cross-Modality Domain Adaptation for Medical Image Segmentation},
  booktitle = {Proceedings of The Thirty-Third Conference on Artificial Intelligence (AAAI)},
  pages     = {865--872},
  year      = {2019},
}
```

## Acknowledgement
Part of the code is revised from the [Tensorflow implementation of CycleGAN](https://github.com/leehomyc/cyclegan-1).

## Note
* The repository is being updated
* Contact: Cheng Chen (chencheng236@gmail.com)

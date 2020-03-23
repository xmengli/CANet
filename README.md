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
* Messidor dataset: http://www.adcis.net/en/third-party/messidor/
* 


## Train
* Modify paramter values in `./config_param.json`
* Run `./main.py` to start the training process

## Evaluate
* Specify the model path and test file path in `./evaluate.py`
* Run `./evaluate.py` to start the evaluation.


## Citation
If you find the code useful for your research, please cite our paper.
```
@article{li2019canet,
  title={CANet: Cross-disease Attention Network for Joint Diabetic Retinopathy and Diabetic Macular Edema Grading},
  author={Li, Xiaomeng and Hu, Xiaowei and Yu, Lequan and Zhu, Lei and Fu, Chi-Wing and Heng, Pheng-Ann},
  journal={IEEE transactions on medical imaging},
  year={2019},
  publisher={IEEE}
}
```

## Acknowledgement
CBAM module is reused from the [Pytorch implementation of CBAM](https://github.com/Jongchan/attention-module).

## Note
* Contact: Xiaomeng Li (xmengli999@gmail.com)

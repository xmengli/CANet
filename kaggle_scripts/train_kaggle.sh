cd ..
FILE="${var}kaggle_drpretrain_multi"
FOLD="${var}fold"
CUDA_VISIBLE_DEVICES='4,5,6,7' python baseline.py /data/xmli/medical/kaggleDR/ kaggle exp/KAGGLE/$FILE --fold_name $FOLD -a resnet50 --gpu 0 -b 400 --lr 0.1 --pretrained --epochs 20000  --decay_epoch 500 --num_class 5 --resume exp/KAGGLE/kaggle_drpretrain/model_converge.pth.tar
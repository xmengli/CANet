cd ..
CUDA_VISIBLE_DEVICES='2' python baseline.py /home/xmli/gpu16_xmli/DR_DEM/ drdme drdme_densenet161_standaug_pretrain_aug --evaluate --resume exp/drdme_densenet161_standaug_dmepretrained_adam_decay150/model_converge.pth.tar  -a densenet161 --gpu 0 -b 40 --lr 0.01 --pretrained --epochs 4000   --decay_epoch 200  --num_class 5

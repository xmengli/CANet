cd ..
CUDA_VISIBLE_DEVICES='0' python baseline.py /home/xmli/gpu14_xmli/REFUGE/ amd amd_densenet161_standaug_pretrain_cosface_decay20 --evaluate --resume  exp/amd_densenet161_standaug_pretrain_cosface_decay20_againnorminit/checkpoint37.pth.tar  -a densenet161 --cosface --gpu 0 -b 40 --lr 0.01 --pretrained --epochs 4000 --decay_epoch 20

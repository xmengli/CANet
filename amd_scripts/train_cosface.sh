cd ..
CUDA_VISIBLE_DEVICES='2' python baseline.py /home/xmli/gpu14_xmli/REFUGE/ amd amd_densenet161_standaug_pretrain_cosface_decay20_againagain --seed 122 -a densenet161 --cosface --gpu 0 -b 40 --lr 0.01 --pretrained --epochs 100 --decay_epoch 20

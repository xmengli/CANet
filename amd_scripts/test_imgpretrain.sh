cd ..
CUDA_VISIBLE_DEVICES='1' python baseline.py /home/xmli/gpu14_xmli/REFUGE/ amd amd_densenet161_standaug_pretrain_explict --evaluate --resume exp/amd_densenet161_standaug_pretrain/model_best.pth.tar -a densenet161 --gpu 0 -b 40 --lr 0.01 --pretrained --epochs 4000   --decay_epoch 200

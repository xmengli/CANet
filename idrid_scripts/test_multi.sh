cd ..
CUDA_VISIBLE_DEVICES='1' python baseline.py /home/xmli/medical/IDRiD/ drdme drdme_multi_densenet161_standaug_pretrain_adam_decay150 -a resnet50 --gpu 0 -b 40 --lr 0.001  --epochs 300   --decay_epoch 150  --num_class 5 --multitask  --crossCBAM --choice both --evaluate --resume /home/xmli/pheng/networkbased/CAN-jointDRDME/ourmodels/CAN_TAD.pth.tar

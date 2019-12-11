cd ..
CUDA_VISIBLE_DEVICES='7' python baseline.py /home/xmli/gpu16_xmli/MESSIDOR/ missidor MESSIDOR/Chen_resnet50_repeat_dropout3times_class2 -a densenet161 --gpu 0 -b 20 --lr 0.001 --pretrained --epochs 300  --decay_epoch 100 --num_class 2  --multitask  --chen  --evaluate --resume exp/MESSIDOR/Chen_resnet50_repeat_dropout3times_class2/checkpoint200.pth.tar

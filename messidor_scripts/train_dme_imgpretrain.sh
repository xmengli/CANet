cd ..
CUDA_VISIBLE_DEVICES='7' python baseline.py /home/xmli/gpu16_xmli/MESSIDOR/ missidor MESSIDOR/dme_resnet50 -a densenet161 --gpu 0 -b 40 --lr 0.001 --pretrained --epochs 300  --decay_epoch 100 --num_class 3  --resume exp/MESSIDOR/dme_resnet50/checkpoint100.pth.tar --evaluate

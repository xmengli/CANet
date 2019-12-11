cd ..
CUDA_VISIBLE_DEVICES='5' python baseline.py /home/xmli/gpu16_xmli/MESSIDOR/ missidor MESSIDOR/dr_resnet50_adam100_2class -a densenet161 --gpu 0 -b 40 --lr 0.001 --pretrained --epochs 300   --decay_epoch 100 --num_class 2

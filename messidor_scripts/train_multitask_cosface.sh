cd ..
CUDA_VISIBLE_DEVICES='2' python baseline.py /home/xmli/gpu16_xmli/MESSIDOR/ missidor MESSIDOR/multi_densenet161_adam100_cosface_m03 -a densenet161 --gpu 0 -b 40 --lr 0.001 --pretrained --epochs 300  --decay_epoch 100 --num_class 4  --multitask  --cosface

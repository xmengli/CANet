cd ..
CUDA_VISIBLE_DEVICES='3' python baseline.py /home/xmli/gpu16_xmli/DR_DEM/ drdme drdme_densenet161_standaug_pretrained_adam_decay150_cosface_s2 -a densenet161 --gpu 0 -b 40 --lr 0.001 --pretrained --epochs 300   --decay_epoch 150  --num_class 5  --cosface --s 20.0 --m 1e-3

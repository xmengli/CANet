cd ..
CUDA_VISIBLE_DEVICES='0,3' python baseline.py /home/xmli/gpu16_xmli/DR_DEM/ drdme drdme_cbam -a densenet161 --gpu 0 -b 40 --lr 0.001 --pretrained  --epochs 300   --decay_epoch 150  --num_class 5

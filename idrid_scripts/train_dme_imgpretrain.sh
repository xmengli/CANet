cd ..
CUDA_VISIBLE_DEVICES='2' python baseline.py /home/xmli/gpu16_xmli/DR_DEM/ drdme drdme_dme_res50_adam100_bam_ranaug -a resnet50 --gpu 0 -b 40 --lr 0.001 --pretrained --epochs 4000   --decay_epoch 100 --num_class 3

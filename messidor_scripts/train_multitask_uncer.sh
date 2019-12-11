cd ..
CUDA_VISIBLE_DEVICES='3' python baseline.py /home/xmli/gpu16_xmli/MESSIDOR/ missidor MESSIDOR/multi_resnet50_simple_2class_CBAM_drop03repeat2_55_uncertainty_retrain2 -a densenet161 --gpu 0 -b 80 --lr 0.001 --pretrained --epochs 300  --decay_epoch 100 --num_class 2  --multitask  --uncer --crossCBAM

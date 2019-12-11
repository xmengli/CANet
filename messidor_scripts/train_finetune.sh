cd ..
FILE="${var}multi_CAN_TSbased_10fold10_1000"
FOLD="${var}fold10"
CUDA_VISIBLE_DEVICES='4' python baseline.py /home/xmli/medical/MESSIDOR/ missidor exp/MESSIDOR/$FILE --fold_name $FOLD -a resnet50 --gpu 0 -b 40 --lr 0.001 --pretrained --epochs 1000  --decay_epoch 500 --num_class 2  --multitask --CAN_TS --resume exp/MESSIDOR/multi_CAN_joint_10fold10_1000/model_converge.pth.tar
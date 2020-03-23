cd ..
max=10
for i in `seq 0 $max`
do
    NUM="${var}$i"
    FILE="${var}/multi_CAN_lamda25_noshare_wpre_simpleaug_10fold$NUM""_1000"
    FOLD="${var}fold$NUM"
    CUDA_VISIBLE_DEVICES='3' python baseline.py ./data/ missidor exp/MESSIDOR/$FILE --fold_name $FOLD -a resnet50 --gpu 0 -b 40 --base_lr 3e-4 --pretrained --epochs 1000  --decay_epoch 500 --num_class 2 --adam --multitask --crossCBAM --choice both --lambda_value 0.25
done

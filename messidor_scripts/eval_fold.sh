cd ..
max=1
for i in `seq 1 $max`
do
    NUM="${var}$i"
    FILE="${var}/multi_produce_loss$NUM""_1000"
    FOLD="${var}fold$NUM"
    CUDA_VISIBLE_DEVICES='3' python baseline.py /home/xmli/medical/MESSIDOR/ missidor exp/MESSIDOR/$FILE --fold_name $FOLD -a resnet50 --gpu 0 -b 40 --base_lr 3e-4 --pretrained --epochs 1000  --decay_epoch 500  --multitask --crossCBAM --choice both --adam --evaluate --resume exp/MESSIDOR/$FILE/model_converge.pth.tar
done
$python average_result.py --filename multi_CAN_lamda25_10fold
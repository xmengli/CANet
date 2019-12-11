cd ..
max=1
for i in `seq 1 $max`
do
    NUM="${var}$i"
    FILE="${var}/multi_produce_loss$NUM""_1000"
    FOLD="${var}fold$NUM"
    CUDA_VISIBLE_DEVICES='2' python baseline.py /home/xmli/medical/MESSIDOR/ missidor exp/MESSIDOR/$FILE --fold_name $FOLD -a resnet50 --gpu 0 -b 40 --base_lr 3e-4 --pretrained --epochs 1000  --decay_epoch 500 --num_class 2 --adam --multitask --crossCBAM --choice both  --lambda_value 0.25   --resume exp/KAGGLE/model_61/model_converge.pth.tar
done
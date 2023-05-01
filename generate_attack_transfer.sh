# MY_CMD="python test.py --dataroot ./place_select_val/$1 --name CAST_model --result_dir $1 --adv"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=$2 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/vgg-pytorch/select_val/0 --name CAST_model --result_dir unattacked_0"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=0 $MY_CMD

# for i in {0..9}
# do
#     MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/attacked_style_image/$i --name CAST_model --result_dir attack_result_$i"
#     echo $MY_CMD
#     CUDA_VISIBLE_DEVICES=1 $MY_CMD
# done   

# for i in {0..9}
# do
#     MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/pytorch-AdaIN/attacked_style_endtoend/result_endtoend_$i --name CAST_model --result_dir AdaAttN_ml_transfer_attack_result_$i"
#     echo $MY_CMD
#     CUDA_VISIBLE_DEVICES=1 $MY_CMD
# done   

# for i in {0..9}
# do
#     MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble/AdaAttN_multilayer_10/$i --name CAST_model --result_dir AdaAttN_ml_transfer_attack_result_$i"
#     echo $MY_CMD
#     CUDA_VISIBLE_DEVICES=6 $MY_CMD
# done 
# for i in {0..9}
# do
#     MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/resized_style_images/$i --name CAST_model --result_dir ensemble_CAST_resize_clean_result_$i --load_size 512 --crop_size 512"
#     echo $MY_CMD
#     CUDA_VISIBLE_DEVICES=4 $MY_CMD
# done 
for i in {1..9}
do
    MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble/ensemble_attack_resize/$i --name CAST_model --result_dir ensemble_resize_attack_result_$i --load_size 512 --crop_size 512"
    echo $MY_CMD
    CUDA_VISIBLE_DEVICES=4 $MY_CMD
done 
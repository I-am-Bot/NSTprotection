# MY_CMD="python test.py --dataroot ./place_select_val/$1 --name CAST_model --result_dir $1 --adv"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=$2 $MY_CMD
for i in {2..9}
do
    # MY_CMD="python test.py --content_dataroot /egr/research-dselab/liyaxin1/unlearnable/content_images_small/ --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/attacked_style_image/$i --name CAST_model --result_dir smallset_attacked_result_$i"
    MY_CMD="python ensemble.py --name CAST_model --style_num $i"
    echo $MY_CMD
    CUDA_VISIBLE_DEVICES=4 $MY_CMD

    MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble/ensemble_attack_resize/$i --name CAST_model --result_dir ensemble_resize_attack_result_$i --load_size 512 --crop_size 512"
    echo $MY_CMD
    CUDA_VISIBLE_DEVICES=4 $MY_CMD
done


# MY_CMD="python test.py --content_dataroot /egr/research-dselab/liyaxin1/unlearnable/content_images_small/ --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble/ensemble_multilayer/3/ --name CAST_model --result_dir ensemble_multilayer_3"

# for i in {0..9}
# do
#     MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/attacked_style_image/$i --name CAST_model --result_dir attack_result_$i"
#     echo $MY_CMD
#     CUDA_VISIBLE_DEVICES=1 $MY_CMD
# done   

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/attacked_result_endtoend_1 --name CAST_model --result_dir attack_endtoend_result_1"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=1 $MY_CMD

# for i in {2..9}
# do
#     MY_CMD="python test_end_to_end.py --adv --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/AdaIN_select_val/$i --name CAST_model --result_dir attacked_result_endtoend_$i"
#     echo $MY_CMD
#     CUDA_VISIBLE_DEVICES=1 $MY_CMD
#     MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/CAST_pytorch/results/attacked_result_endtoend_$i --name CAST_model --result_dir attack_endtoend_result_$i"
#     echo $MY_CMD
#     CUDA_VISIBLE_DEVICES=1 $MY_CMD
# done   

# MY_CMD="python ../compare.py"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=1 $MY_CMD
# echo "sleep"
# sleep 15000
# echo "sleep done"

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_15000/0 --name CAST_model --result_dir ensemble_result_15000_0"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_15000/1 --name CAST_model --result_dir ensemble_result_15000_1"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_15000/2 --name CAST_model --result_dir ensemble_result_15000_2"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_15000/3 --name CAST_model --result_dir ensemble_result_15000_3"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_15000/4 --name CAST_model --result_dir ensemble_result_15000_4"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_15000/5 --name CAST_model --result_dir ensemble_result_15000_5"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_15000/6 --name CAST_model --result_dir ensemble_result_15000_6"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_15000/7 --name CAST_model --result_dir ensemble_result_15000_7"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_15000/8 --name CAST_model --result_dir ensemble_result_15000_8"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_15000/9 --name CAST_model --result_dir ensemble_result_15000_9"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD



# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1000/0 --name CAST_model --result_dir ensemble_result_15000_0"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1000/1 --name CAST_model --result_dir ensemble_result_1000_1"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1000/2 --name CAST_model --result_dir ensemble_result_1000_2"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1000/3 --name CAST_model --result_dir ensemble_result_1000_3"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1000/4 --name CAST_model --result_dir ensemble_result_1000_4"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1000/5 --name CAST_model --result_dir ensemble_result_1000_5"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1000/6 --name CAST_model --result_dir ensemble_result_1000_6"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1000/7 --name CAST_model --result_dir ensemble_result_1000_7"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1000/8 --name CAST_model --result_dir ensemble_result_1000_8"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1000/9 --name CAST_model --result_dir ensemble_result_1000_9"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD




# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1/0 --name CAST_model --result_dir ensemble_result_1_0"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1/1 --name CAST_model --result_dir ensemble_result_1_1"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1/2 --name CAST_model --result_dir ensemble_result_1_2"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1/3 --name CAST_model --result_dir ensemble_result_1_3"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1/4 --name CAST_model --result_dir ensemble_result_1_4"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1/5 --name CAST_model --result_dir ensemble_result_1_5"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1/6 --name CAST_model --result_dir ensemble_result_1_6"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1/7 --name CAST_model --result_dir ensemble_result_1_7"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1/8 --name CAST_model --result_dir ensemble_result_1_8"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD

# MY_CMD="python test.py --content_dataroot ./datasets/transfer/testA --style_dataroot /egr/research-dselab/liyaxin1/unlearnable/ensemble_adv_1/9 --name CAST_model --result_dir ensemble_result_1_9"
# echo $MY_CMD
# CUDA_VISIBLE_DEVICES=5 $MY_CMD
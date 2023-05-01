from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image

def img_reshape(img):
    img = Image.open(img).convert('RGB')
    img = np.asarray(img)
    return img

# images = os.listdir('./output_original/')
# images_non = os.listdir('./output_adv_non/')
# images_reverse = os.listdir('./output_non_reverse/')

style = []
image_original_array = []
image_attack_non_array = []
image_attack_reverse_array = []

for i in range(1, 20):
    style.append(img_reshape('./datasets/demo2/testB/' + str(i) + '.jpg'))
    image_original_array.append(img_reshape('./results/CAST_model/cornell_' + str(i) + '.png'))
    image_attack_non_array.append(img_reshape('./results/CAST_model/cornell_attack_' + str(i) + '.png'))
    # image_attack_reverse_array.append(img_reshape('./results/1_attack_' + str(i) + '_reverse_' + str(i) + '.jpg'))
    
fig, axes = plt.subplots(nrows=6, ncols=10, figsize=(100, 100), squeeze=False)
plt.rcParams["savefig.bbox"] = 'tight'

for i in range(10):
    axes[0, i].imshow(style[i])
    axes[1, i].imshow(image_original_array[i])
    axes[2, i].imshow(image_attack_non_array[i])
    if i != 9:
        axes[3, i].imshow(style[i+10])
        axes[4, i].imshow(image_original_array[i+10])
        axes[5, i].imshow(image_attack_non_array[i+10])
    # axes[3, i].imshow(image_attack_reverse_array[i])
    axes[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axes[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axes[2, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    axes[3, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axes[4, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axes[5, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    # axes[3, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
input("save")
plt.savefig('cornell_result')

# fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(100, 100), squeeze=False)
# plt.rcParams["savefig.bbox"] = 'tight'

# output = []
# for i in range(1, 11):
#     images = []
#     for j in range(1, 11):
#         if i != j:
#             images.append(img_reshape('./output_adv_target/lenna_stylized_attack_' + str(i) + '_' + str(j) + '.jpg'))
#         else:
#             images.append(img_reshape('./output_original/lenna_stylized_' + str(i) + '.jpg'))
#     output.append(images)

# for i in range(10):
#     for j in range(10):
#         if i != j:
#             axes[i, j].imshow(output[i][j])
#             axes[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#         else:
#             axes[i, j].imshow(output[i][j])
#             axes[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#             axes[i, j].spines['bottom'].set_color('red')
#             axes[i, j].spines['top'].set_color('red')
#             axes[i, j].spines['left'].set_color('red')
#             axes[i, j].spines['right'].set_color('red')
#             axes[i, j].spines['bottom'].set_linewidth(10)
#             axes[i, j].spines['top'].set_linewidth(10)
#             axes[i, j].spines['left'].set_linewidth(10)
#             axes[i, j].spines['right'].set_linewidth(10)            
# plt.savefig('lenna_target_result')



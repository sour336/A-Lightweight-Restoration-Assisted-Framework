import os
# 读取数据集的目录信息,生成数据的组织文件

# 训练集
blur_dir = './Dataset/eval/img'
deblur_dir = './Dataset/eval/lab'
mask_dir = './Dataset/eval/mask'
txt_name = './eval_list_mask'               # 保存的名称

blur_img_list =[]
deblur_img_list = []
mask_img_list = []
# 遍历img目录下的图像
for filename in os.listdir(blur_dir):
    print(filename)
    # 获取文件的完整路径
    filepath = os.path.join(blur_dir, filename)
    # 判断是否为文件
    if os.path.isfile(filepath):
        blur_img_list.append(filepath)
# 遍历lab目录下的图像
for filename in os.listdir(deblur_dir):
    # 获取文件的完整路径
    filepath = os.path.join(deblur_dir, filename)
    # 判断是否为文件
    if os.path.isfile(filepath):
        deblur_img_list.append(filepath)

for filename in os.listdir(mask_dir):
    # 获取文件的完整路径
    filepath = os.path.join(mask_dir, filename)
    # 判断是否为文件
    if os.path.isfile(filepath):
        mask_img_list.append(filepath)

# 先检查数据数量上是否有问题
assert len(blur_img_list)==len(deblur_img_list)
assert len(blur_img_list)==len(mask_img_list)

# 将数据组织成为一个list,使用文件保存
with open(txt_name, 'w') as file:
    for a,b,c in zip(blur_img_list,deblur_img_list,mask_img_list):
        img_pairs = "{} {} {}\n".format(a, b, c)
        file.write(img_pairs)
file.close()

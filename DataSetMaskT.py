import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 自定义数据集类,目的是是使用DataLoader可以对数据进行批处理、打乱和并行加载
# 需要创建一个继承自 torch.utils.DataSet.Dataset 的自定义数据集类。这个类应该至少包含两个方法：__len__ 返回数据集的大小，__getitem__ 返回给定索引的数据。

class Dataset(torch.utils.data.Dataset):
    """加载和处理训练集(含有blur和deblur对应图片,加入随机裁剪),将图片转化为张量"""
    def __init__(self, img_list, crop_size=(256, 256)):
        """
        Args:
            img_list:  图像文件列表
            crop_size: 表示裁剪后的图像大小
        """
        super(type(self), self).__init__()
        self.img_list = img_list
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        根据给定的训练索引idx获取对应的图像数据
        Args:
            idx:    给定的训练数据索引(可能是多个)
        Returns:
            batch:  键值对,将256x256、128x128和64x64大小的三张图像转换为批量的张量形式(张量大小在[-1,1]之间)
        """
        # 得到idx
        blurry_img_name = self.img_list[idx].split(' ')[-3]
        clear_img_name = self.img_list[idx].split(' ')[-2]
        mask_img_name = self.img_list[idx].split(' ')[-1]

        blurry_img = Image.open(blurry_img_name)
        clear_img = Image.open(clear_img_name)
        mask_img = Image.open(mask_img_name)
        # 断言判断模糊图像和清晰图像的size是否相同
        assert blurry_img.size == clear_img.size
        # assert blurry_img.size == mask_img.size

        blurry_img = blurry_img.resize((320,320),resample=Image.BILINEAR)
        clear_img = clear_img.resize((320,320),resample=Image.BILINEAR)
        mask_img = mask_img.resize((320,320),resample=Image.BILINEAR)

        img_size = blurry_img.size

        img240 = blurry_img.resize((img_size[0] * 3 // 4, img_size[1] * 3 // 4), resample=Image.BILINEAR)
        img160 = blurry_img.resize((img_size[0] // 2, img_size[1] // 2), resample=Image.BILINEAR)

        label240 = clear_img.resize((img_size[0] * 3 // 4, img_size[1] * 3 // 4), resample=Image.BILINEAR)
        label160 = clear_img.resize((img_size[0] // 2, img_size[1] // 2), resample=Image.BILINEAR)

        mask240 = mask_img.resize((img_size[0] * 3 // 4, img_size[1] * 3 // 4), resample=Image.BILINEAR)
        mask160 = mask_img.resize((img_size[0] // 2, img_size[1] // 2), resample=Image.BILINEAR)

        blurry_img = self.to_tensor(blurry_img)
        img240 = self.to_tensor(img240)
        img160 = self.to_tensor(img160)

        clear_img = self.to_tensor(clear_img)
        label240 = self.to_tensor(label240)
        label160 = self.to_tensor(label160)

        mask_img = self.to_tensor(mask_img)
        mask240 = self.to_tensor(mask240)
        mask160 = self.to_tensor(mask160)

        # 将3个size,模糊与清晰的六组图像数据放入字典batch中
        batch = {'img320': blurry_img, 'img240': img240, 'img160': img160,
                 'label320': clear_img, 'label240': label240, 'label160': label160,
                 'mask320': mask_img, 'mask240': mask240, 'mask160': mask160,
                 'mask320_t': mask_img, 'mask240_t': mask240, 'mask160_t': mask160}

        key = ['mask320','mask240','mask160']
        for k in batch:
            if k not in key:
                batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]



        return batch

class ValDataset(torch.utils.data.Dataset):
    """加载和处理训练集(含有blur和deblur对应图片,不加入随机裁剪),将图片转化为张量"""
    def __init__(self, img_list):
        """
        Args:
            img_list:  图像文件列表
        """
        super(type(self), self).__init__()
        self.img_list = img_list
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        根据给定的训练索引idx获取对应的图像数据
        Args:
            idx:    给定的训练数据索引(可能是多个)
        Returns:
            batch:  键值对,将256x256、128x128和64x64大小的三张图像转换为批量的张量形式(张量大小在[-1,1]之间)
        """
        # 得到idx
        blurry_img_name = self.img_list[idx].split(' ')[-3]
        clear_img_name = self.img_list[idx].split(' ')[-2]
        mask_img_name = self.img_list[idx].split(' ')[-1]

        blurry_img = Image.open(blurry_img_name)
        clear_img = Image.open(clear_img_name)
        mask_img = Image.open(mask_img_name)
        # 断言判断模糊图像和清晰图像的size是否相同
        assert blurry_img.size == clear_img.size
        # assert blurry_img.size == mask_img.size

        blurry_img = blurry_img.resize((320,320),resample=Image.BILINEAR)
        clear_img = clear_img.resize((320,320),resample=Image.BILINEAR)
        mask_img = mask_img.resize((320,320),resample=Image.BILINEAR)

        img_size = blurry_img.size

        img240 = blurry_img.resize((img_size[0] * 3 // 4, img_size[1] * 3 // 4), resample=Image.BILINEAR)
        img160 = blurry_img.resize((img_size[0] // 2, img_size[1] // 2), resample=Image.BILINEAR)

        label240 = clear_img.resize((img_size[0] * 3 // 4, img_size[1] * 3 // 4), resample=Image.BILINEAR)
        label160 = clear_img.resize((img_size[0] // 2, img_size[1] // 2), resample=Image.BILINEAR)

        mask240 = mask_img.resize((img_size[0] * 3 // 4, img_size[1] * 3 // 4), resample=Image.BILINEAR)
        mask160 = mask_img.resize((img_size[0] // 2, img_size[1] // 2), resample=Image.BILINEAR)

        blurry_img = self.to_tensor(blurry_img)
        img240 = self.to_tensor(img240)
        img160 = self.to_tensor(img160)

        clear_img = self.to_tensor(clear_img)
        label240 = self.to_tensor(label240)
        label160 = self.to_tensor(label160)

        mask_img = self.to_tensor(mask_img)
        mask240 = self.to_tensor(mask240)
        mask160 = self.to_tensor(mask160)

        # 将3个size,模糊与清晰的六组图像数据放入字典batch中
        batch = {'img320': blurry_img, 'img240': img240, 'img160': img160,
                 'label320': clear_img, 'label240': label240, 'label160': label160,
                 'mask320': mask_img, 'mask240': mask240, 'mask160': mask160,
                 'mask320_t': mask_img, 'mask240_t': mask240, 'mask160_t': mask160}

        key = ['mask320', 'mask240', 'mask160']
        for k in batch:
            if k not in key:
                batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]


        return batch
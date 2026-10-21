import torch
import numpy as np
import os
from model.osrnet import OSRNet  # 导入基本网络
import torchvision.transforms as transforms
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


def load_img(file):
    """读取filepath处的图片"""
    occlusion_img = Image.open(file)
    occlusion_img = occlusion_img.resize((320,320),resample=Image.BILINEAR)
    size = occlusion_img.size
    to_tesnor = transforms.ToTensor()

    # 原规模,0.5规模,0.25规模
    img1 = occlusion_img
    img2 = img1.resize((size[0] *3 // 4, size[1] *3 // 4), resample=Image.BILINEAR)
    img3 = img2.resize((size[0] // 2, size[1] // 2), resample=Image.BILINEAR)
    # 转换为tensor,并增加第0维
    img1 = torch.unsqueeze(to_tesnor(img1), 0)
    img2 = torch.unsqueeze(to_tesnor(img2), 0)
    img3 = torch.unsqueeze(to_tesnor(img3), 0)
    # 组合成为batch字典输出
    batch = {'img1.0': img1, 'img0.75': img2, 'img0.5': img3}
    for k in batch:
        batch[k] = batch[k] * 2 - 1.0  # in range [-1,1]
    return batch


def restoration(input):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input_file = input

    # 加载网络结构
    net = torch.nn.DataParallel(OSRNet(xavier_init_all='xavier_init_all'))
    checkpoints = torch.load(r'./weight/Student/best.pth',map_location=device)
    net.load_state_dict(checkpoints)
    net.to(device)
    net.eval()

    # 图片处理
    batch = load_img(input_file)
    with torch.inference_mode():
        _, _, _, _, _, _, db320_tensor, _, _ = net(batch['img1.0'], batch['img0.75'], batch['img0.5'])

        # 删去batchsize维度,限制到Imaes的Tensor(-1,1)
        db320 = torch.squeeze(db320_tensor, 0).clamp(-1, 1)
        db320 = (db320 + 1) / 2

        # 从tensor转换到image
        to_pil = transforms.ToPILImage()
        new_img = to_pil(db320)

        # 计算像素级差异
        diff = torch.abs(db320_tensor - batch['img1.0'].to(db320_tensor.device))
        diff_mean = diff.float().mean(dim=1, keepdim=True)

        # 每张图片分别计算空间最小值和最大值
        diff_min = diff_mean.amin(dim=(2, 3), keepdim=True)
        diff_max = diff_mean.amax(dim=(2, 3), keepdim=True)

        # 给分母设置下限，避免除以零
        denominator = (diff_max - diff_min).clamp_min(1e-6)
        diff_norm = (diff_mean - diff_min) / denominator

        # 生成二值遮挡掩码
        mask = (diff_norm > 0.08).float()

        mask_np = mask.squeeze().cpu().numpy() * 255
        mask_img = Image.fromarray(mask_np.astype(np.uint8))

    return input_file, new_img, mask_img
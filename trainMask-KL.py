import torch
import numpy as np
import config                               # 导入参数的设置
from model.osrnet import OSRNet
from model.osrnetT import OSRNetT  # 导入基本网络
from DataSetMaskT import Dataset, ValDataset     # 引入训练和验证数据集加载/处理方法
from tqdm import tqdm                       # 在命令行界面中显示进度条
from time import time
import sys

log10 = np.log(10)  # log10
MAX_DIFF = 2  # 图片可能的最大像素值(图片的张量值在[-1,1]之间),在峰值信噪比(PSNR)计算时需要
device = torch.device('cuda')


def FFT_L1_loss(pred, gt):
    pred_nor = (pred+1)/2
    gt_nor = (gt+1)/2

    pred_g = pred_nor.mean(dim=1,keepdim=True)
    gt_g = gt_nor.mean(dim=1,keepdim=True)

    f_pred = torch.fft.rfft2(pred_g)
    f_gt = torch.fft.rfft2(gt_g)
    loss = (f_pred.real - f_gt.real).abs().mean() + (f_pred.imag - f_gt.imag).abs().mean()

    return loss


def compute_loss(db320, db240, db160, batch, istrain, fea_s, fea_t, db320_t, epoch):
    assert db320.shape[0] == batch['label320'].shape[0]

    loss = 0
    loss1 = 0
    loss2 = 0
    loss3 = 0
    loss4 = 0

    loss_mse1 = 0
    loss_mse2 = 0
    loss_mse3 = 0
    loss_mse1 += mse(db320, batch['label320'])
    loss_mse2 += mse(db240, batch['label240'])
    loss_mse3 += mse(db160, batch['label160'])

    loss1 += mse(db320, batch['label320'])
    loss1 += mse(db240, batch['label240'])
    loss1 += mse(db160, batch['label160'])
    if istrain == True:
        # loss2 += ((batch['mask320'] * (db320 - batch['label320'])**2).sum(dim=(2,3)) / (
        #         batch['mask320'].sum(dim=(2,3)) + 1e-8)).mean()

        # loss2 += ((batch['mask320'] * (db320 - db320_t) ** 2).sum(dim=(2, 3)) / (
        #             batch['mask320'].sum(dim=(2, 3)) + 1e-8)).mean()

        n = (epoch+1)//120
        loss2_1 = ((batch['mask320'] * (db320 - batch['label320'])**2).sum(dim=(2,3)) / (
                batch['mask320'].sum(dim=(2,3)) + 1e-8)).mean()
        loss2_2 = ((batch['mask320'] * (db320 - db320_t) ** 2).sum(dim=(2, 3)) / (
                batch['mask320'].sum(dim=(2, 3)) + 1e-8)).mean()
        loss2 = (1-n) * loss2_2 + n * loss2_1

    loss3 += FFT_L1_loss(db320, batch['label320'])
    for i in range(6):
        for j in range(3):
            fea_s[i][j] = torch.nn.functional.normalize(fea_s[i][j], p=2, dim=(1))
            fea_t[i][j] = torch.nn.functional.normalize(fea_t[i][j], p=2, dim=(1))
            loss4 += mse(fea_s[i][j], fea_t[i][j].detach())

        loss += loss1 + loss2 + 0.001*loss3 + 0.01 * loss4
    # 峰值信噪比(PSNR)
    psnr = 10 * torch.log(MAX_DIFF ** 2 / loss_mse1) / log10
    return {'total': loss, 'psnr': psnr, 'mse1':loss_mse1, 'mse2':loss_mse2, 'mse3':loss_mse3,'FFT':loss3, 'KL':loss4+loss2}


def backward(loss, optimizer):
    """
    Arg:
        loss:       键值对,损失函数计算值,包含mse(均方误差),psnr(最大规模图片上的峰值信噪比)
        optimizer:  优化器
    Return:

    """
    optimizer.zero_grad()  # 将优化器中的梯度清零
    loss['total'].backward()  # 对 MSE 损失进行反向传播
    # torch.nn.utils.clip_grad_norm_(net.module.convlstm.parameters(), 3)  # 对整个模型的梯度进行裁剪,这里的3是裁剪梯度的阈值
    optimizer.step()  # 更新模型的参数
    return


def set_learning_rate(optimizer, epoch):
    """
    使用了学习率衰减策略,根据epoch设置优化器optimizer的学习率,即周期越大,学习率越低
    Arg:
        optimizer: 优化器
        epoch:     当前训练的周期数
    """
    if epoch < 10:
        optimizer.param_groups[0]['lr'] = config.train['learning_rate'] * 0.01
    if 10 <= epoch < 110:
        optimizer.param_groups[0]['lr'] = config.train['learning_rate'] * 0.1 ** ((epoch - 9) // 100)
    if epoch >= 110:
        optimizer.param_groups[0]['lr'] = config.train['learning_rate'] * 0.01

if __name__ == "__main__":
    # 1.读入数据集,并放入数据加载器中
    # 读入训练集和测试集数据,其中.read() 读取文件的全部内容为一个字符串,.strip() 用于去除字符串两端的空格和换行符等空白字符。
    train_img_list = open(config.train['train_img_list'], 'r').read().strip().split('\n')
    val_img_list = open(config.train['val_img_list'], 'r').read().strip().split('\n')
    # 将数据集放入DataLoader(数据加载器)中
    train_dataset = Dataset(train_img_list)
    val_dataset = ValDataset(val_img_list)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train['batch_size'],
                                                   shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train['val_batch_size'],
                                                 shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    # 2.均方误差
    mse = torch.nn.MSELoss().cuda()

    # 3.网络
    S_net = torch.nn.DataParallel(OSRNet(xavier_init_all=config.net['xavier_init_all'])).cuda()
    T_net = torch.nn.DataParallel(OSRNetT(xavier_init_all=config.net['xavier_init_all'])).cuda()
    T_checkpoints = torch.load("./checkpoints/119-T.pth")
    T_net.load_state_dict(T_checkpoints)
    T_net.eval()
    for p in T_net.parameters():
        p.requires_grad = False

    # 如果使用之前已训练的网络参数
    if config.train['if_use_pretrained_model']:
        checkpoints = torch.load(config.train['used_params_dir'])
        S_net.load_state_dict(checkpoints)

    # 4.优化器
    assert config.train['optimizer'] in ['Adam', 'SGD']
    if config.train['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(S_net.parameters(), lr=config.train['learning_rate'],
                                     weight_decay=config.loss['weight_l2_reg'])
    if config.train['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(S_net.parameters(), lr=config.train['learning_rate'],
                                    weight_decay=config.loss['weight_l2_reg'], momentum=config.train['momentum'],
                                    nesterov=config.train['nesterov'])

    # 定义一些用于记录训练的参数
    train_loss_log_list = []        # 用于记录log记录时训练集的损失值
    val_loss_log_list = []          # 用于记录log记录时的验证集损失值
    first_val = True
    t = time()

    # 定义一些用于验证集上最优的参数和模型
    best_val_psnr = 0               # 记录目前为止在验证集上达到的最佳PSNR(峰值信噪比)值
    best_net = None                 # 验证过程中达到最佳 PSNR 时的模型
    best_optimizer = None           # 验证过程中达到最佳 PSNR 时的优化器
    a = 0

    for epoch in tqdm(range(config.train['num_epochs']), file=sys.stdout, desc=str(config.train['num_epochs'])+' epoches'):
        # 根据当前 epoch 设置学习率
        set_learning_rate(optimizer, epoch)

        # 训练
        for step, batch in enumerate(train_dataloader):
            a = a+1
            # print(a)
            # 这里的batch是键值对,但是值的第0维度是bathsize
            # 将batch数据移到GPU上,不需要计算梯度
            for k in batch:
                batch[k] = batch[k].cuda()
                batch[k].requires_grad = False
            # 得到网络预测结果
            S_f1, S_f2, S_f3, S_f4, S_f5, S_f6, S_db320, S_db240, S_db160 = S_net(batch['img320'], batch['img240'],
                                                                                 batch['img160'])

            with torch.no_grad():
                T_f1, T_f2, T_f3, T_f4, T_f5, T_f6, T_db320, _, _ = T_net(batch['img320'], batch['img240'],
                                                                          batch['img160'],
                                                                          batch['mask320_t'], batch['mask240_t'],
                                                                          batch['mask160_t'])
            # 计算损失
            loss = compute_loss(S_db320, S_db240, S_db160, batch, istrain=True,
                                fea_s=[S_f1, S_f2, S_f3, S_f4, S_f5, S_f6],
                                fea_t=[T_f1, T_f2, T_f3, T_f4, T_f5, T_f6],
                                db320_t= T_db320, epoch=epoch)

            # 反向传播和网络参数更新
            backward(loss, optimizer)

            # 将loss从gpu移动到cpu上
            for k in loss:
                loss[k] = float(loss[k].cpu().detach().numpy())
            # 记录训练的损失值
            train_loss_log_list.append({k: loss[k] for k in loss})


        # 验证(间隔log_epoch个周期验证一次)
        if first_val or epoch % config.train['log_epoch'] == config.train['log_epoch'] - 1:
            first_val = False
            # 验证时不需要记录梯度
            with torch.no_grad():
                for step, batch in enumerate(val_dataloader):
                    for k in batch:
                        batch[k] = batch[k].cuda()
                        batch[k].requires_grad = False
                    S_f1, S_f2, S_f3, S_f4, S_f5, S_f6, S_db320, S_db240, S_db160 = S_net(batch['img320'],
                                                                                         batch['img240'],
                                                                                         batch['img160'])

                    T_f1, T_f2, T_f3, T_f4, T_f5, T_f6, T_db320, _, _ = T_net(batch['img320'], batch['img240'],
                                                                              batch['img160'],
                                                                              batch['mask320_t'], batch['mask240_t'],
                                                                              batch['mask160_t'])
                    loss = compute_loss(S_db320, S_db240, S_db160, batch, istrain=True,
                                        fea_s=[S_f1, S_f2, S_f3, S_f4, S_f5, S_f6],
                                        fea_t=[T_f1, T_f2, T_f3, T_f4, T_f5, T_f6],
                                        db320_t= T_db320, epoch=epoch)
                    for k in loss:
                        loss[k] = float(loss[k].cpu().detach().numpy())
                    val_loss_log_list.append({k: loss[k] for k in loss})
                # 计算了训练损失(MSE和)的平均值
                train_loss_log_dict = {k: float(np.mean([dic[k] for dic in train_loss_log_list])) for k in
                                       train_loss_log_list[0]}
                val_loss_log_dict = {k: float(np.mean([dic[k] for dic in val_loss_log_list])) for k in
                                     val_loss_log_list[0]}


                # PSNR的值越大越好
                if best_val_psnr < val_loss_log_dict['psnr']:
                    best_val_psnr = val_loss_log_dict['psnr']   # 保存最优的PSNR值
                    best_net = S_net.state_dict()                 # 更新最优模型参数

                torch.save(S_net.state_dict(), './checkpoints/' + str(epoch) + '.pth')

                # 将训练集和测试集的损失列表清空
                train_loss_log_list.clear()
                val_loss_log_list.clear()

                tt = time()
                log_msg = ""
                log_msg += "epoch {} , {:.2f} imgs/s".format(epoch, (
                            config.train['log_epoch'] * len(train_dataloader) * config.train['batch_size'] + len(
                        val_dataloader) * config.train['val_batch_size']) / (tt - t))

                log_msg += " | train : "
                for idx, k_v in enumerate(train_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',')
                log_msg += "  | eval : "
                for idx, k_v in enumerate(val_loss_log_dict.items()):
                    k, v = k_v
                    if k == 'acc':
                        log_msg += "{} {:.3%} {}".format(k, v, ',')
                    else:
                        log_msg += "{} {:.5f} {}".format(k, v, ',' if idx < len(val_loss_log_list) - 1 else '')
                tqdm.write(log_msg, file=sys.stdout)
                sys.stdout.flush()
                t = time()


    print("最优PSNR为:", best_val_psnr)
    torch.save(best_net, config.train['save_params_dir'])
# net是关于神经网络的一些配置
net = {}
net['xavier_init_all'] = True                   # 是否使用 Xavier 初始化所有权重

# loss关于损失函数的一些配置
loss = {}
loss['weight_l2_reg'] = 0.0                     # 权重 L2 正则化的系数。

# 训练过程的一些配置参数
train = {}
train['train_img_list'] = './train_list_mask'        # 训练图像列表文件的路径
train['val_img_list'] = './eval_list_mask'           # 验证图像列表文件的路径。
train['batch_size'] = 4                                 # 训练时的批量大小。
train['val_batch_size'] = 1                             # 验证时的批量大小。
train['num_epochs'] = 120                              # 训练的总轮数。
train['log_epoch'] = 1                                # 每隔多少轮输出一次日志
train['optimizer'] = 'Adam'                             # 优化器的选择，这里是 'Adam'
train['learning_rate'] = 1e-4                           # 学习率的设置
# SGD优化器的参数设置
train['momentum'] = 0.9                                 # 对于 SGD 优化器的动量参数
train['nesterov'] = True                                # 对于 SGD 优化器是否使用 Nesterov 动量
# 模型保存的设置
train['if_use_pretrained_model'] = False                            # 是否使用已经训练的参数
train['used_params_dir'] = ''         # 已经训练的模型参数的文件路径
train['save_teacher_params_dir'] = './weight/Teacher/bestT.pth'         # 将要保存模型的参数的文件路径
train['save_student_params_dir'] = './weight/Student/best.pth'
train['teacher_weight'] = './weight/Teacher/bestT.pth'

# 测试过程中的一些参数配置
test = {}
test['model_params'] = './weight/Student/best.pth'
test['input_dir'] = './TestData/input'
test['output_dir']= './TestData/output'                    # 输出去模糊后测试图像的目录
test['output_prefix'] = 're_'                        # 新图片的前缀名
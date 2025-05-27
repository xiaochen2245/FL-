import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    ### 客户端
    parser.add_argument('--num_users', type=int, default=10) #客户端数量
    parser.add_argument('--frac', type=float, default=1) # 选择的客户端比例
    parser.add_argument('--partial_data', type=float, default=0.1) # 每个客户端使用的数据比例
    
    ### 模型和特征大小
    parser.add_argument('--models', type=str, default='cnnbn') # cnn (MNIST), cnnbn (FMNIST), mlp 使用的模型类型 
    parser.add_argument('--output_channel', type=int, default=1) # 输出通道数，默认值为1
    parser.add_argument('--img_size', type=int, default=32) # 图像大小，默认值为32
    parser.add_argument('--orig_img_size', type=int, default=32) # 原始图像大小，默认值为32

    ### 数据集
    parser.add_argument('--dataset', type=str, default='fmnist') # stl10, cifar10, svhn, fmnist, mnist, emnist 使用的数据集
    parser.add_argument('--noniid', action='store_true') # 默认：否 是否使用非独立同分布（Non-IID）数据
    parser.add_argument('--dir_param', type=float, default=0.3) # 用于控制Non-IID程度的参数，默认值为0.3
    parser.add_argument('--num_classes', type=int, default=10) # 数据集的类别数，默认值为10

    ### 优化器
    parser.add_argument('--bs', type=int, default=64) # 批量大小
    parser.add_argument('--local_bs', type=int, default=64) # 本地批量大小
    parser.add_argument('--momentum', type=float, default=0) # 优化器的动量参数
    parser.add_argument('--weight_decay', type=float, default=0) #权重衰减（L2正则化）

    ### 可重复性
    parser.add_argument('--rs', type=int, default=0) # 随机种子
    parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments") # 实验次数
    parser.add_argument('--device_id', type=str, default='0') # 使用的GPU设备ID

    ### FLWG
    parser.add_argument('--wu_epochs', type=int, default=20)
    parser.add_argument('--freeze_FE', type=bool, default=True)

    ### FLWG
    parser.add_argument('--gen_wu_epochs', type=int, default=100) # 生成器的预热轮数，默认值为100
    parser.add_argument('--epochs', type=int, default=50) # 总训练轮数，默认值为50
    parser.add_argument('--local_ep', type=int, default=5) # 每个客户端的本地训练轮数，默认值为5
    parser.add_argument('--local_ep_gen', type=int, default=1) # 使用生成样本训练主网络的本地轮数，默认值为1
    parser.add_argument('--gen_local_ep', type=int, default=5) # 训练生成器的本地轮数，默认值为5
    parser.add_argument('--aid_by_gen', type=int, default=1) # 是否使用生成器辅助训练
    parser.add_argument('--freeze_gen', type=int, default=1) # 是否冻结生成器 GAN: False
    parser.add_argument('--avg_FE', type=int, default=1) # 是否使用LG-FedAvg算法
    parser.add_argument('--only_gen', type=int, default=0) # 是否仅使用生成器进行训练

    ### logging
    parser.add_argument('--sample_test', type=int, default=10) # 测试样本的数量，默认值为10
    parser.add_argument('--save_imgs', type=bool, default=False) # 是否保存生成的图像，本地训练的生成器图像
    parser.add_argument('--wandb', type=int, default=0) # 是否使用Weights & Biases进行日志记录
    parser.add_argument('--wandb_proj_name', type=str, default='FLWG') #Weights & Biases项目名称
    parser.add_argument('--name', type=str, default='FLWG') # 实验名称
    parser.add_argument('--plot_interval', type=int, default=1) 
    parser.add_argument('--save_dir', type=str, default='imgs/') # 保存图像的目录

    parser.add_argument('--gen_model', type=str, default='vae') # vae, gan, ddpm 使用的生成模型类型
    ### VAE 参数
    parser.add_argument('--latent_size', type=int, default=16) # VAE的潜在空间大小，默认值为16
    ### GAN 参数
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient") # GAN中Adam优化器的第一个动量衰减率
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient") # GAN中Adam优化器的第二个动量衰减率
    parser.add_argument('--gan_lr', type=float, default=0.0002) # GAN的学习率
    parser.add_argument('--latent_dim', type=int, default=100) # GAN的潜在空间维度
    ### DDPM 参数
    parser.add_argument('--n_feat', type=int, default=128) # DDPM的特征数量 128效果还行 256更好但会更慢
    parser.add_argument('--n_T', type=int, default=200) # DDPM的时间步数 还可以再大一点 400 500
    parser.add_argument('--guide_w', type=float, default=0.0) # DDPM的引导权重 0, 0.5, 2

    ### 目标网络
    parser.add_argument('--lr', type=float, default=1e-1) # 目标网络的学习率

    ### FedProx 参数
    parser.add_argument('--fedprox', type=bool, default=False) # 是否使用FedProx算法
    parser.add_argument('--mu', type=float, default=1e-2) # FedProx的μ参数

    ### AvgKD ####
    parser.add_argument('--avgKD', type=bool, default=False) # 是否使用平均知识蒸馏

    args = parser.parse_args()
    args.device = 'cuda:' + args.device_id
    
    if args.dataset == 'fmnist' or args.dataset == 'mnist':
        args.gen_wu_epochs = 100
        args.epochs = 50
        if not args.freeze_gen:
            args.gen_wu_epochs = args.gen_wu_epochs - args.epochs #如果数据集是fmnist或mnist，则调整生成器的预热轮数和总训练轮数

    if 'ddpm' in args.gen_model:
        args.name = args.name + 'w' + str(args.guide_w) #如果生成模型是ddpm，则在实验名称中添加引导权重
                
    return args

import numpy as np
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict
#加载pickle文件

class DatasetSplit(Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
#创建一个子数据集，包含指定索引的数据

def dict_iid(dataset, num_users):

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
#生成**独立同分布（IID）**的数据划分

def mnist_noniid(dataset, num_users, seed):
    """
    从 MNIST 数据集中采样非独立同分布的客户端数据
    """
    np.random.seed(seed)

    num_shards, num_imgs = 200, 300 # 2 (class) x 100 (users), 2 x 300 (imgs) for each client
    # {0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy() # targets

    # 标签排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] # label-wise sort
    idxs = idxs_labels[0,:]

    # 划分并分配
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return

def cifar_noniid(args, dataset):
    """
    从 CIFAR 数据集抽取 50000 个非独立同分布的客户端数据样本
    """
    np.random.seed(args.rs)

    num_shards, num_imgs = args.num_users * args.class_per_each_client, int(50000/args.num_users/args.class_per_each_client)
    # {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 5000, 5: 5000, 6: 5000, 7: 5000, 8: 5000, 9: 5000}
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(args.num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)
    
    # 标签排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] # label-wise sort
    idxs = idxs_labels[0,:]

    # 划分并分配
    for i in range(args.num_users):
        rand_set = set(np.random.choice(idx_shard, args.class_per_each_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def noniid_dir(args, beta, dataset):
    '''
    狄利克雷分布
    较小的β>0 分割更加不平衡
    '''

    np.random.seed(args.rs)
    random.seed(args.rs)
    min_size = 0
    min_require_size = 10
    
    N = len(dataset)
    net_dataidx_map = {}
    labels = np.array(dataset.targets)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(args.num_users)]
        for k in range(args.num_classes): 
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, args.num_users)) # 所有用户使用同一个β
            proportions = np.array([p * (len(idx_j) < N / args.num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])


    for j in range(args.num_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    
    return net_dataidx_map


def getDataset(args):
    if args.dataset == 'mnist' and 'mlp' in args.models:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        
        dataset_train = datasets.MNIST('/datasets/chen/.data/mnist', train=True, download=True, transform=transform_train)
        dataset_test = datasets.MNIST('/datasets/chen/.data/mnist', train=False, download=True, transform=transform_test)

    elif args.dataset == 'mnist' and 'cnn' in args.models:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.orig_img_size),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.orig_img_size),
        ])
        
        dataset_train = datasets.MNIST('/datasets/chen/.data/mnist', train=True, download=True, transform=transform_train)
        dataset_test = datasets.MNIST('/datasets/chen/.data/mnist', train=False, download=True, transform=transform_test)

    elif args.dataset == 'fmnist' and 'mlp' in args.models:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        
        dataset_train = datasets.FashionMNIST('/datasets/chen/.data/fmnist', train=True, download=True, transform=transform_train)
        dataset_test = datasets.FashionMNIST('/datasets/chen/.data/fmnist', train=False, download=True, transform=transform_test)

    elif args.dataset == 'fmnist' and 'cnn' in args.models:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.orig_img_size),
            transforms.Normalize((0.5,), (0.5,)), 
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.orig_img_size),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        
        dataset_train = datasets.FashionMNIST('/datasets/chen/.data/fmnist', train=True, download=True, transform=transform_train)
        dataset_test = datasets.FashionMNIST('/datasets/chen/.data/fmnist', train=False, download=True, transform=transform_test)
                
    elif args.dataset =='cifar10':
        ## CIFAR数据集 包含10类物体的彩色图像数据集
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('/datasets/chen/.data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('/datasets/chen/.data/cifar', train=False, download=True, transform=transform_test)

    elif args.dataset == 'svhn':
        ### SVHN数据集 街景门牌号数据集，包含门牌号图像
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])

        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
            ])
        dataset_train = datasets.SVHN('/datasets/chen/.data/svhn', split='train', download=True, transform=transform_train)
        dataset_test = datasets.SVHN('/datasets/chen/.data/svhn', split='test', download=True, transform=transform_test)

    elif args.dataset == 'stl10':
        ### STL10数据集 包含10类物体的图像数据集，常用于无监督学习
        transform_train = transforms.Compose([
                        transforms.RandomCrop(96, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.2471, 0.2435, 0.2616])
                    ])
        transform_test = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465],
                                            [0.2471, 0.2435, 0.2616])
                ])
        dataset_train = datasets.STL10('/datasets/chen/.data/stl10', split='train', download=True, transform=transform_train)
        dataset_test = datasets.STL10('/datasets/chen/.data/stl10', split='test', download=True, transform=transform_test)

    return dataset_train, dataset_test
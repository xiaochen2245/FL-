'''
https://github.com/znxlwm/pytorch-MNIST-CelebA-cGAN-cDCGAN/tree/master

Conv2dtranspose
https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11
https://velog.io/@hayaseleu/Transposed-Convolutional-Layer%EC%9D%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80
output shape of conv2dtranspose k=kernel, s=stride, p=padding
1d input length l -> l + (s-1)*(l-1) + 2*(k-p-1)
then Conv by k*k kernel, stride=1
output shape: l + (s-1)*(l-1) + 2*(k-p-1) - k + 1
l：输入特征图的长度（高度或宽度）
k：卷积核的大小
s：步幅
p：填充
torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
def output_shape(l,k,s,p):
    return l + (s-1)*(l-1) + 2*(k-p-1) - k + 1
DCGAN-F (args.output_channel x 32 x 32)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import one_hot


# G(z)s
class generator(nn.Module):
    # initializers
    def __init__(self, args, d=128):
        super(generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0) # in ch, out ch, kernel size, stride, padding
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)
        # self.deconv5 = nn.ConvTranspose2d(d*2, args.output_channel, 4, 2, 1)
        
        self.args = args

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
                
        x = torch.tanh(self.deconv4(x))
        
        return x

    def sample_image(self, args, sample_num=0):
        with torch.no_grad():
        
            z_ = torch.randn((sample_num, 100)).view(-1, 100, 1, 1).to(args.device)
            y_ = (torch.rand(sample_num, 1) * 10).type(torch.LongTensor).squeeze()

            onehot = torch.zeros(10, 10)
            onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1) # 10 x 10 eye matrix

            y_label_ = onehot[y_]
            y_label_ = y_label_.to(args.device)
            gen_imgs = self.forward(z_, y_label_)
            
            # c = torch.randint(10, (args.batch_size, )).to(args.device) # MAX_NUM, (SIZE, )
            one_c = one_hot(y_, args.num_classes).to(args.device)

            return gen_imgs, one_c

    def sample_image_4visualization(self, sample_num):
        
        with torch.no_grad():
            z_ = torch.randn((sample_num, 100)).view(-1, 100, 1, 1).to(self.args.device)
            labels = torch.arange(0,10) # context for us just cycles throught the mnist labels
            labels = labels.repeat(int(sample_num/labels.shape[0]))

            onehot = torch.zeros(10, 10)
            onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1) # 10 x 10 eye matrix

            y_label_ = onehot[labels]
            y_label_ = y_label_.to(self.args.device)
            gen_imgs = self.forward(z_, y_label_)

            return gen_imgs


class discriminator(nn.Module):
    # initializers
    def __init__(self, args, d=128):
        super(discriminator, self).__init__()
        # self.conv1_1 = nn.Conv2d(1, int(d/2), 4, 2, 1) # 1 64
        self.conv1_1 = nn.Conv2d(args.output_channel, int(d/2), 4, 2, 1) # 1 64
        self.conv1_2 = nn.Conv2d(10, int(d/2), 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1) # 128 256
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1) # 256 512
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0) # 512 1

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label): # input 1 x 32 x 32
        x = F.leaky_relu(self.conv1_1(input), 0.2) # 128 x 16 x 16
        y = F.leaky_relu(self.conv1_2(label), 0.2) # 128 x 16 x 16
        x = torch.cat([x, y], 1) # 256 x 16 x  16
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2) # 512 x 8 x 8
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2) # 1024 x 4 x 4
        x = torch.sigmoid(self.conv4(x)) #

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
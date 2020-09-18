import torch
from torch import nn
from torch.nn import functional as F



class Res_GCN(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True, sub_sample=True,):
        super(Res_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None


        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))


    def forward(self, f):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size = f.size(0)

        g_f = self.g(f).view(batch_size, self.inter_channels, -1)
        g_f = g_f.permute(0, 2, 1)

        theta_f = self.theta(f).view(batch_size, self.inter_channels, -1)
        theta_f = theta_f.permute(0, 2, 1)
        phi_f = self.phi(f).view(batch_size, self.inter_channels, -1)
        A = torch.matmul(theta_f, phi_f)
        A_div_C = F.softmax(A, dim=-1)

        y = torch.matmul(A_div_C, g_f)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, * f.size()[2:])
        W_y = self.W(y)
        f_star = W_y + f

        return f_star

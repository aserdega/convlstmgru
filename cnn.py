import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class ConvEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super(ConvEncoder, self).__init__()
        self.cnn_encoder = nn.Sequential(
                                nn.Conv2d(1,32,3,stride=1,padding=1),
                                nn.BatchNorm2d(32),
                                nn.ELU(),
                                nn.Conv2d(32,64,3,stride=2,padding=1),
                                nn.BatchNorm2d(64),
                                nn.ELU(),
                                nn.Conv2d(64,128,3,stride=2,padding=1),
                                nn.BatchNorm2d(128),
                                nn.ELU(),
                                nn.Conv2d(128,out_dim,3,stride=2,padding=1),
                                nn.Tanh()
                                )

        self.out_dim = out_dim

    def forward(self, input):
        output = self.cnn_encoder(input)
        return output


class ConvDecoder(nn.Module):
    def __init__(self, b_size=32, inp_dim=64):
        super(ConvDecoder, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(inp_dim,128,3,stride=2, padding=1)
        self.bn1    = nn.BatchNorm2d(128)
        self.dconv2 = nn.ConvTranspose2d(128,64,3,stride=2, padding=1)
        self.bn2    = nn.BatchNorm2d(64)
        self.dconv3 = nn.ConvTranspose2d(64,32,3,stride=2, padding=1)
        self.bn3    = nn.BatchNorm2d(32)
        self.dconv4 = nn.ConvTranspose2d(32,1,3,stride=1, padding=1)

        self.size1  = torch.Size([b_size * 10, 128, 16, 16])
        self.size2  = torch.Size([b_size * 10, 64, 32, 32])
        self.size3  = torch.Size([b_size * 10, 32, 64, 64])

        self.inp_dim = inp_dim

    def forward(self, input):
        h1 = self.bn1(self.dconv1(input, self.size1))
        a1 = F.elu(h1)
        h2 = self.bn2(self.dconv2(a1, self.size2))
        a2 = F.elu(h2)
        h3 = self.bn3(self.dconv3(a2, self.size3))
        a3 = F.elu(h3)
        h4 = self.dconv4(a3)
        return h4
'''

class ConvEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super(ConvEncoder, self).__init__()
        self.cnn_encoder = nn.Sequential(
                                nn.Conv2d(1,16,3,stride=1,padding=1),
                                nn.BatchNorm2d(16),
                                nn.ELU(),
                                nn.Conv2d(16,32,3,stride=2,padding=1),
                                nn.BatchNorm2d(32),
                                nn.ELU(),
                                nn.Conv2d(32,64,3,stride=1,padding=1),
                                nn.BatchNorm2d(64),
                                nn.ELU(),
                                nn.Conv2d(64,out_dim,3,stride=2,padding=1),
                                nn.Tanh()
                                )

        self.out_dim = out_dim
        self.device  = 'cpu'

    def cuda(self, device='cuda'):
        super(ConvEncoder, self).cuda(device)
        self.device = device

    def forward(self, input):
        output = self.cnn_encoder(input)
        return output


class ConvDecoder(nn.Module):
    def __init__(self, b_size=32, inp_dim=64):
        super(ConvDecoder, self).__init__()
        self.dconv1 = nn.ConvTranspose2d(inp_dim,64,3,stride=2, padding=1)
        self.bn1    = nn.BatchNorm2d(64)
        self.dconv2 = nn.ConvTranspose2d(64,32,3,stride=1, padding=1)
        self.bn2    = nn.BatchNorm2d(32)
        self.dconv3 = nn.ConvTranspose2d(32,16,3,stride=2, padding=1)
        self.bn3    = nn.BatchNorm2d(16)
        self.dconv4 = nn.ConvTranspose2d(16,1,3,stride=1, padding=1)

        self.size1  = torch.Size([b_size * 10, 64, 32, 32])
        self.size2  = torch.Size([b_size * 10, 16, 64, 64])

        self.inp_dim = inp_dim

    def forward(self, input):
        h1 = self.bn1(self.dconv1(input, self.size1))
        a1 = F.elu(h1)
        h2 = self.bn2(self.dconv2(a1))
        a2 = F.elu(h2)
        h3 = self.bn3(self.dconv3(a2, self.size2))
        a3 = F.elu(h3)
        h4 = self.dconv4(a3)
        return h4

class MCConvEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super(MCConvEncoder, self).__init__()
        self.encoder = nn.Sequential(
                                nn.Conv2d(1,64,5,stride=1,padding=2),
                                nn.BatchNorm2d(64),
                                nn.ELU(),
                                nn.MaxPool2d(2,stride=2),
                                nn.Conv2d(64,128,5,stride=1,padding=2),
                                nn.BatchNorm2d(128),
                                nn.ELU(),
                                nn.MaxPool2d(2,stride=2),
                                nn.Conv2d(128,out_dim,7,stride=1,padding=3),
                                nn.Tanh(),
                                nn.MaxPool2d(2,stride=2),
                                )

        self.out_dim = out_dim

    def forward(self, input):
        output = self.encoder(input)
        return output


class MCConvDecoder(nn.Module):
    def __init__(self, b_size=32, inp_dim=256):
        super(MCConvDecoder, self).__init__()
        self.decoder = nn.Sequential(
                                nn.Upsample(scale_factor=2),
                                nn.ConvTranspose2d(inp_dim,128,7,stride=1,padding=3),
                                nn.BatchNorm2d(128),
                                nn.ELU(),
                                nn.Upsample(scale_factor=2),
                                nn.ConvTranspose2d(128,64,5,stride=1,padding=2),
                                nn.BatchNorm2d(64),
                                nn.ELU(),
                                nn.Upsample(scale_factor=2),
                                nn.ConvTranspose2d(64,1,5,stride=1,padding=2),
                                )

        self.inp_dim = inp_dim

    def forward(self, input):
        output = self.decoder(input)
        return output


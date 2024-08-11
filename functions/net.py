import torch
import torch.nn as nn
import torch.nn.functional as F

import vgg_encoder

################################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect')
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.leakyrelu(x)
        return x

class DecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DecodeBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.c1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.c2 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
        self.c3 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
        self.c4 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
        self.c5 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
        self.c6 = ConvBlock(out_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        x = self.upsample(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = self.c6(x)
        return x
################################################################################
class SR_v3(nn.Module):
    def __init__(self):
        super(SR_v3, self).__init__()
        # sub models
        self.vgg_3 = vgg_encoder.relu_3_4()
        self.vgg_2 = vgg_encoder.relu_2_2()
        self.vgg_1 = vgg_encoder.relu_1_2()
        
        for param in self.vgg_3.parameters():
            param.requires_grad = False
        for param in self.vgg_2.parameters():
            param.requires_grad = False
        for param in self.vgg_1.parameters():
            param.requires_grad = False
        
        self.depth  = 3
        self.scale  = 4

        # VGG Decoder Starts
        self.p0_1 = ConvBlock(256, 256, kernel_size=5, stride=1, padding=2)
        self.p0_2 = ConvBlock(256, 256, kernel_size=5, stride=1, padding=2)
        self.p0_3 = ConvBlock(256, 256, kernel_size=5, stride=1, padding=2)
        self.p0_4 = ConvBlock(256, 256, kernel_size=5, stride=1, padding=2)

        # SR
        self.d4 = DecodeBlock(256, 256, kernel_size=5, stride=1, padding=2)
        self.d3 = DecodeBlock(256, 256, kernel_size=5, stride=1, padding=2)
        # Decode
        self.d2 = DecodeBlock(256, 128, kernel_size=5, stride=1, padding=2)
        self.d1 = DecodeBlock(128, 64, kernel_size=5, stride=1, padding=2)

        # to Image
        self.e1 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e2 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e3 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e4 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e5 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e6 = ConvBlock(64, 32, kernel_size = 3, stride=1, padding=1)
        self.e7 = ConvBlock(32, 16, kernel_size = 3, stride=1, padding=1)
        self.end = ConvBlock(16, 3, kernel_size = 3, stride=1, padding=1)

    def forward(self, x):
        lr_img  = x # for residual.
        lr2x = F.interpolate(x, scale_factor=2, mode='bicubic')
        lr4x = F.interpolate(x, scale_factor=4, mode='bicubic')

        x = self.vgg_3(x) # 256, (width//4, heigth//4)

        # Convolution
        x = self.p0_1(x)
        x = self.p0_2(x)
        x = self.p0_3(x)
        x = self.p0_4(x)
        x = x + self.vgg_3(lr_img) # 256, (width//4, heigth//4)

        x4 = x # loss - p4

        # Block 4 (SR) #
        x = self.d4(x)
        x = x + self.vgg_3(lr2x)

        x3 = x # loss - p3

        # Block 3 (SR) #
        x = self.d3(x)
        x = x + self.vgg_3(lr4x)

        x2 = x # loss - p2

        # Block 2 #
        x = self.d2(x)
        x = x + self.vgg_2(lr4x)

        x1 = x # loss - p1

        # Block 1 #
        x = self.d1(x)
        x = x + self.vgg_1(lr4x)
        x0 = x # loss - p0

        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        x = self.e5(x)
        x = self.e6(x)
        x = self.e7(x)
        x = self.end(x)
        x = x + lr4x

        return x, (x0, x1, x2, x3, x4)

################################################################################
# advanced4

class SR_v4(nn.Module):
    def __init__(self):
        super(SR_v4, self).__init__()
        # sub models
        self.vgg_4 = vgg_encoder.relu_4_4()
        self.vgg_3 = vgg_encoder.relu_3_4()
        self.vgg_2 = vgg_encoder.relu_2_2()
        self.vgg_1 = vgg_encoder.relu_1_2()
        self.depth  = 4
        self.scale  = 4

        for param in self.vgg_4.parameters():
            param.requires_grad = False
        for param in self.vgg_3.parameters():
            param.requires_grad = False
        for param in self.vgg_2.parameters():
            param.requires_grad = False
        for param in self.vgg_1.parameters():
            param.requires_grad = False

        # VGG Decoder Starts
        self.p0_1 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.p0_2 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.p0_3 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.p0_4 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)

        self.s2 = DecodeBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.s1 = DecodeBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.d3 = DecodeBlock(512, 256, kernel_size=5, stride=1, padding=2)
        self.d2 = DecodeBlock(256, 128, kernel_size=5, stride=1, padding=2)
        self.d1 = DecodeBlock(128, 64, kernel_size=5, stride=1, padding=2)

        # to Image
        self.e1 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e2 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e3 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e4 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e5 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e6 = ConvBlock(64, 32, kernel_size = 3, stride=1, padding=1)
        self.e7 = ConvBlock(32, 16, kernel_size = 3, stride=1, padding=1)
        self.end = ConvBlock(16, 3, kernel_size = 3, stride=1, padding=1)

        # res 4x image

    def forward(self, x ):
        lr_img  = x # for residual.
        lr2x = F.interpolate(x, scale_factor=2, mode='bicubic')
        lr4x = F.interpolate(x, scale_factor=4, mode='bicubic')

        x = self.vgg_4(x)

        # Convolution
        x = self.p0_1(x)
        x = self.p0_2(x)
        x = self.p0_3(x)
        x = self.p0_4(x)
        x = x + self.vgg_4(lr_img) 

        x5 = x 

        x = self.s2(x)
        x = x + self.vgg_4(lr2x)

        x4 = x 

        x = self.s1(x)
        x = x + self.vgg_4(lr4x)

        x3 = x 

        x = self.d3(x)
        x = x + self.vgg_3(lr4x)

        x2 = x 

        x = self.d2(x)
        x = x + self.vgg_2(lr4x)

        x1 = x 

        x = self.d1(x)
        x = x + self.vgg_1(lr4x)
        x0 = x # loss - p0

        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        x = self.e5(x)
        x = self.e6(x)
        x = self.e7(x)
        x = self.end(x)
        x = x + lr4x

        return x, (x0, x1, x2, x3, x4, x5)
################################################################################
# advanced 5

class SR_v5(nn.Module):
    def __init__(self):
        super(SR_v5, self).__init__()
        # sub models
        self.vgg_5 = vgg_encoder.relu_5_4()
        self.vgg_4 = vgg_encoder.relu_4_4()
        self.vgg_3 = vgg_encoder.relu_3_4()
        self.vgg_2 = vgg_encoder.relu_2_2()
        self.vgg_1 = vgg_encoder.relu_1_2()
        self.depth  = 5
        self.scale  = 4

        for param in self.vgg_5.parameters():
            param.requires_grad = False
        for param in self.vgg_4.parameters():
            param.requires_grad = False
        for param in self.vgg_3.parameters():
            param.requires_grad = False
        for param in self.vgg_2.parameters():
            param.requires_grad = False
        for param in self.vgg_1.parameters():
            param.requires_grad = False

        # VGG Decoder Starts
        self.p0_1 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.p0_2 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.p0_3 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.p0_4 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)

        self.s2 = DecodeBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.s1 = DecodeBlock(512, 512, kernel_size=5, stride=1, padding=2)

        self.d4 = DecodeBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.d3 = DecodeBlock(512, 256, kernel_size=5, stride=1, padding=2)
        self.d2 = DecodeBlock(256, 128, kernel_size=5, stride=1, padding=2)
        self.d1 = DecodeBlock(128, 64, kernel_size=5, stride=1, padding=2)

        # to Image
        self.e1 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e2 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e3 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e4 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e5 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e6 = ConvBlock(64, 32, kernel_size = 3, stride=1, padding=1)
        self.e7 = ConvBlock(32, 16, kernel_size = 3, stride=1, padding=1)
        self.end = ConvBlock(16, 3, kernel_size = 3, stride=1, padding=1)

        # res 4x image

    def forward(self, x ):
        lr_img  = x # for residual.
        lr2x = F.interpolate(x, scale_factor=2, mode='bicubic')
        lr4x = F.interpolate(x, scale_factor=4, mode='bicubic')

        x = self.vgg_5(x)

        # Convolution
        x = self.p0_1(x)
        x = self.p0_2(x)
        x = self.p0_3(x)
        x = self.p0_4(x)
        x = x + self.vgg_5(lr_img) 

        x6 = x 

        x = self.s2(x)
        x = x + self.vgg_5(lr2x)

        x5 = x 

        x = self.s1(x)
        x = x + self.vgg_5(lr4x)

        x4 = x 

        x = self.d4(x)
        x = x + self.vgg_4(lr4x)

        x3 = x

        x = self.d3(x)
        x = x + self.vgg_3(lr4x)

        x2 = x 

        x = self.d2(x)
        x = x + self.vgg_2(lr4x)

        x1 = x 

        x = self.d1(x)
        x = x + self.vgg_1(lr4x)
        x0 = x # loss - p0

        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        x = self.e5(x)
        x = self.e6(x)
        x = self.e7(x)
        x = self.end(x)
        x = x + lr4x

        return x, (x0, x1, x2, x3, x4, x5, x6)

#################################################################################

class SR_v4_x8(nn.Module):
    def __init__(self):
        super(SR_v4_x8, self).__init__()
        # sub models
        self.vgg_4 = vgg_encoder.relu_4_4()
        self.vgg_3 = vgg_encoder.relu_3_4()
        self.vgg_2 = vgg_encoder.relu_2_2()
        self.vgg_1 = vgg_encoder.relu_1_2()
        self.depth  = 4
        self.scale  = 8

        # VGG Decoder Starts
        self.p0_1 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.p0_2 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.p0_3 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.p0_4 = ConvBlock(512, 512, kernel_size=5, stride=1, padding=2)

        self.s3 = DecodeBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.s2 = DecodeBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.s1 = DecodeBlock(512, 512, kernel_size=5, stride=1, padding=2)
        self.d3 = DecodeBlock(512, 256, kernel_size=5, stride=1, padding=2)
        self.d2 = DecodeBlock(256, 128, kernel_size=5, stride=1, padding=2)
        self.d1 = DecodeBlock(128, 64, kernel_size=5, stride=1, padding=2)

        # to Image
        self.e1 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e2 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e3 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e4 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e5 = ConvBlock(64, 64, kernel_size = 3, stride=1, padding=1)
        self.e6 = ConvBlock(64, 32, kernel_size = 3, stride=1, padding=1)
        self.e7 = ConvBlock(32, 16, kernel_size = 3, stride=1, padding=1)
        self.end = ConvBlock(16, 3, kernel_size = 3, stride=1, padding=1)

        # res 4x image

    def forward(self, x ):
        lr_img  = x # for residual.
        lr2x = F.interpolate(x, scale_factor=2, mode='bicubic')
        lr4x = F.interpolate(x, scale_factor=4, mode='bicubic')
        lr8x = F.interpolate(x, scale_factor=8, mode='bicubic')

        x = self.vgg_4(x)

        # Convolution
        x = self.p0_1(x)
        x = self.p0_2(x)
        x = self.p0_3(x)
        x = self.p0_4(x)
        x = x + self.vgg_4(lr_img) 

        x6 = x 

        x = self.s3(x)
        x = x + self.vgg_4(lr2x)
        
        x5 = x

        x = self.s2(x)
        x = x + self.vgg_4(lr4x)

        x4 = x 

        x = self.s1(x)
        x = x + self.vgg_4(lr8x)

        x3 = x 

        x = self.d3(x)
        x = x + self.vgg_3(lr8x)

        x2 = x 

        x = self.d2(x)
        x = x + self.vgg_2(lr8x)

        x1 = x 

        x = self.d1(x)
        x = x + self.vgg_1(lr8x)
        x0 = x # loss - p0

        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        x = self.e5(x)
        x = self.e6(x)
        x = self.e7(x)
        x = self.end(x)
        x = x + lr8x

        return x, (x0, x1, x2, x3, x4, x5, x6)
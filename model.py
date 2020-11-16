
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

'''alias'''
SN = nn.utils.spectral_norm

def Conv2d(use_sn, *args, **kwargs):
    if use_sn: return SN(nn.Conv2d(*args, **kwargs))
    else:      return nn.Conv2d(*args, **kwargs)

def Norm2d(name, *args, **kwargs):
    assert name in ['bn', 'in']
    if name == 'bn': return nn.BatchNorm2d(*args, **kwargs)
    elif name == 'in': return nn.InstanceNorm2d(*args, **kwargs)

def Act(name, *args, **kwargs):
    assert name in ['relu', 'lrelu']
    if name == 'relu': return nn.ReLU(*args, **kwargs)
    elif name == 'lrelu': return nn.LeakyReLU(0.2, *args, **kwargs)

'''conv norm activate'''
class ConvBlock(nn.Module):
    def __init__(self,
        use_sn, norm_name, act_name, *args, **kwargs
    ):
        super().__init__()
        self.block = nn.Sequential(
            Conv2d(use_sn, *args, **kwargs),
            Norm2d(norm_name, args[1]),
            Act(act_name)
        )
    def forward(self, x):
        return self.block(x)

'''Unet'''

class UNet(nn.Module):
    def __init__(self,
        depth, in_channels, out_channels, mid_channels,
        use_sn, norm_name, act_name, up_mode, down_mode
    ):
        super().__init__()

        convb_func = functools.partial(
            ConvBlock, use_sn, norm_name, act_name, bias=False if norm_name=='bn' else True
        )
        self.input = convb_func(in_channels, out_channels, 3, padding=1)
        self.encoders = nn.ModuleList([convb_func(out_channels, mid_channels, 3, padding=1)])
        for _ in range(depth-2):
            self.encoders.append(convb_func(mid_channels, mid_channels, 3, padding=1))
        self.bottom = convb_func(mid_channels, mid_channels, 3, padding=2, dilation=2)
        self.decoders = nn.ModuleList()
        for _ in range(depth-2):
            self.decoders.append(convb_func(mid_channels*2, mid_channels, 3, padding=1))
        self.decoders.append(convb_func(mid_channels*2, out_channels, 3, padding=1))

        self.upsample = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True)
        if down_mode == 'max': self.down = nn.MaxPool2d(2, 2, ceil_mode=True)
        elif down_mode == 'avg': self.down = nn.AvgPool2d(2)
        else: raise Exception('no such down sample method')

    def forward(self, x):
        h = self.input(x)
        hin = h

        hs = []
        for module in self.encoders:
            h = module(h)
            hs.append(h)
            h = self.down(h)
        
        h = self.bottom(h)

        for module, h_down in zip(self.decoders, hs[::-1]):
            h = self.upsample(h)
            h = module(torch.cat([h, h_down], dim=1))

        return h + hin

class UNetFlat(nn.Module):
    def __init__(self,
        depth, in_channels, out_channels, mid_channels,
        use_sn, norm_name, act_name
    ):
        super().__init__()

        conv_func = functools.partial(ConvBlock, use_sn, norm_name, act_name, bias=False if norm_name=='bn' else True)
        self.input = conv_func(in_channels, out_channels, 3, padding=1)
        self.encoders = nn.ModuleList([conv_func(out_channels, mid_channels, 3, padding=1)])
        for _ in range(depth-2):
            self.encoders.append(conv_func(mid_channels, mid_channels, 3, padding=1))
        self.bottom = conv_func(mid_channels, mid_channels, 3, padding=2, dilation=2)
        self.decoders = nn.ModuleList()
        for _ in range(depth-2):
            self.decoders.append(conv_func(mid_channels*2, mid_channels, 3, padding=1))
        self.decoders.append(conv_func(mid_channels*2, out_channels, 3, padding=1))

    def forward(self, x):

        h = self.input(x)
        hin = h

        hs = []
        for module in self.encoders:
            h = module(h)
            hs.append(h)

        h = self.bottom(h)

        for module, h_enc in zip(self.decoders, hs[::-1]):
            h = module(torch.cat([h, h_enc], dim=1))

        return h + hin

class U2Net(nn.Module):
    def __init__(self,
        depth=6, in_channels=1, out_channels=1, channels=32,
        use_sn=False, norm_name='bn', act_name='relu',
        up_mode='bilinear', down_mode='avg', img_output=True
    ):
        super().__init__()

        # define layers
        conv_func = functools.partial(Conv2d, use_sn, bias=False if norm_name=='bn' else True)
        unet_func = functools.partial(UNet,
            use_sn=use_sn, norm_name=norm_name, act_name=act_name,
            up_mode=up_mode, down_mode=down_mode)
        unetf_func = functools.partial(UNetFlat,
            use_sn=use_sn, norm_name=norm_name, act_name=act_name)
        unet_depth = depth + 1

        # creat encoders
        self.encoders = nn.ModuleList([unet_func(unet_depth, in_channels, channels*2, channels)])
        unet_depth -= 1
        channels *= 2
        for _ in range(depth-3):
            self.encoders.append(unet_func(unet_depth, channels, channels*2, channels//2))
            unet_depth -= 1
            channels *= 2
        self.encoders.append(unetf_func(unet_depth, channels, channels, channels//2))
        # creat bottom layer and side conv
        self.bottom = unetf_func(unet_depth, channels, channels, channels//2)
        self.sides = nn.ModuleList([conv_func(channels, out_channels, 3, padding=1)])
        # creat decoders and side convs
        self.decoders = nn.ModuleList([unetf_func(unet_depth, channels*2, channels, channels//2)])
        self.sides.append(conv_func(channels, out_channels, 3, padding=1))
        for _ in range(depth-3):
            self.decoders.append(unet_func(unet_depth, channels*2, channels//2, channels//4))
            self.sides.append(conv_func(channels//2, out_channels, 3, padding=1))
            unet_depth += 1
            channels = channels // 2
        self.decoders.append(unet_func(unet_depth, channels*2, channels//2, out_channels))
        self.sides.append(conv_func(channels//2, out_channels, 3, padding=1))

        # final output layer
        self.out_conv = nn.Sequential(
            conv_func(depth*out_channels, out_channels, 1),
            nn.Tanh() if img_output else nn.Identity()
        )

        # up/down sampling
        self.upsample = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True)
        self.mode = up_mode
        if down_mode == 'max': self.downsample = nn.MaxPool2d(2, 2, ceil_mode=True)
        elif down_mode == 'avg': self.downsample = nn.AvgPool2d(2)
        else: raise Exception('no such down sample method')

    def forward(self, x):
        _, _, H, W = x.size()

        h = x

        hs = []
        # encode
        for module in self.encoders:
            h = module(h)
            hs.append(h)
            h = self.downsample(h)

        # bottom layer
        h = self.bottom(h)
        hs_side = [h]

        # decode
        for module, h_down in zip(self.decoders, hs[::-1]):
            h = self.upsample(h)
            h = module(torch.cat([h, h_down], dim=1))
            hs_side.append(h)
        
        hs = []
        # accumulate each upsampled decoders output
        for module, h_side in zip(self.sides, hs_side):
            hs.append(self._upsample_to(module(h_side), (H, W)))

        # final conv
        out = self.out_conv(torch.cat(hs, dim=1))

        return out

    def _upsample_to(self, tensor, size):
        return F.interpolate(tensor, size=size, mode=self.mode, align_corners=True)

class Discriminator(nn.Module):
    def __init__(self,
        in_channels=1, channels=32, n_layers=3,
        use_sn=False, norm_name='bn', act_name='lrelu'
    ):
        super().__init__()

        conv_func = functools.partial(Conv2d, use_sn, bias=False if norm_name=='bn' else True)
        convb_func = functools.partial(
            ConvBlock, use_sn, norm_name, act_name, bias=False if norm_name=='bn' else True
        )

        self.discriminator = nn.ModuleList([
            conv_func(in_channels, channels, 4, stride=2, padding=1),
            Norm2d(norm_name, channels)
        ])
        for _ in range(1, n_layers):
            self.discriminator.append(convb_func(channels, channels*2, 4, stride=2, padding=1))
            channels *= 2
        
        self.discriminator.extend([
            convb_func(channels, channels*2, 4, padding=1),
            conv_func(channels*2, 1, 4, padding=1)
        ])

    def forward(self, x):
        for module in self.discriminator:
            x = module(x)
        return x

def init_weights_normal(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1., 0.02)
        if not m.bias == None:
            m.bias.data.fill_(0.)

def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if not m.bias == None:
            m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.xavier_normal_(m.weight)
        if not m.bias == None:
            m.bias.data.fill_(0.)

if __name__ == "__main__":
    img = torch.randn(32, 1, 256, 256)
    # kwargs = dict(depth=4, in_channels=1, out_channels=1, mid_channels=32,
    #     use_sn=True, norm_name='bn', act_name='relu')
    # module = UNetFlat(**kwargs)

    g = U2Net()
    # print(g)
    output = g(img)
    print(output.size())

    d = Discriminator()
    prob = d(img)
    print(prob.size())

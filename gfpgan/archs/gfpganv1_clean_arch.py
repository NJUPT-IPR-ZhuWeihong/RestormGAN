import math
import random
import torch
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn
from torch.nn import functional as F

from .stylegan2_clean_arch import StyleGAN2GeneratorClean


class StyleGAN2GeneratorCSFT(StyleGAN2GeneratorClean):
    """StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).

    It is the clean version without custom compiled CUDA extensions used in StyleGAN2.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(self, out_size, num_style_feat=512, num_mlp=8, channel_multiplier=2, narrow=1, sft_half=False):
        super(StyleGAN2GeneratorCSFT, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow)
        self.sft_half = sft_half

    def forward(self,
                styles,
                conditions,
                input_is_latent=False,
                noise=None,
                randomize_noise=True,
                truncation=1,
                truncation_latent=None,
                inject_index=None,
                return_latents=False):
        """Forward function for StyleGAN2GeneratorCSFT.

        Args:
            styles (list[Tensor]): Sample codes of styles.
            conditions (list[Tensor]): SFT conditions to generators.
            input_is_latent (bool): Whether input is latent style. Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
            truncation (float): The truncation ratio. Default: 1.
            truncation_latent (Tensor | None): The truncation latent tensor. Default: None.
            inject_index (int | None): The injection index for mixing noise. Default: None.
            return_latents (bool): Whether to return style latents. Default: False.
        """
        # style codes -> latents with Style MLP layer
        #   U-Net中间得到的style code通过MLP转为潜在信息W
        #   输入styles {Tensor:(1,16,512)}
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [getattr(self.noises, f'noise{i}') for i in range(self.num_layers)]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(truncation_latent + truncation * (style - truncation_latent))
            styles = style_truncation
        # get style latents with injection
        if len(styles) == 1:
            #   inject_index = 16
            inject_index = self.num_latent

            #   styles[0].ndim = 3
            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                #   latent = {Tensor:(1,16,512)}
                #   潜在信息W
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        '''
        A.这里是StyleGAN的知识，先初始化一个常量的输入
        B.self.style_conv1():
        （1）完成style对输入的加权调制操作
        （2）完成噪声noise对输入的偏置操作，同时为噪声添加了权重
        （3）此外对输入又添加了一个偏置bias
        C.self.to_rgb1():
        （1）通过style对输入x进行手动加权二维卷积操作且上采样
        （2）添加一个偏置bias
        （3）若skip为None，则结束操作；若skip为sth，则进行上采样且再添加一个skip偏置
        '''
        #   out = {Tensor:(1,512,4,4)}
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        '''
        偶数的self.style_convs进行上采样，奇数的self.style_convs不进行采样
        总的来说就是添加了style手动加权操作且有各种噪声偏置的复杂单个二维卷积操作
        '''
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            #   out = {Tensor:(1,512,8,8)}
            #   这里的out的是常量初始化后进行两次卷积后的结果
            #   然后利用latent信息对out进行有上采样的style卷积
            out = conv1(out, latent[:, i], noise=noise1)

            '''
            SFT部分
            '''
            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                '''
                out被平等地分成两个部分，一个部分为out_same，另一个部分为out_sft
                out_same不作操作直接保留，out_sft通过condition_scale与condition_shift的值进行加权求和
                完成后再将out_same与out_sft进行concatenate，从而保留一定的身份信息
                '''
                if self.sft_half:  # only apply SFT to half of the channels
                    #   out_same = {Tensor:(1,256,8,8)}
                    #   out_sft = {Tensor:(1,256,8,8)}
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    out = out * conditions[i - 1] + conditions[i]

            '''
            StylyGAN的style加权与噪声部分，各进行7次
            '''
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class ResBlock(nn.Module):
    """Residual block with bilinear upsampling/downsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        mode (str): Upsampling/downsampling mode. Options: down | up. Default: down.
    """

    #   in_channels = 32, out_channels = 64
    def __init__(self, in_channels, out_channels, mode='down'):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == 'down':
            self.scale_factor = 0.5
        elif mode == 'up':
            self.scale_factor = 2

    def forward(self, x):
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        # upsample/downsample
        out = F.interpolate(out, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        # skip
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        skip = self.skip(x)
        out = out + skip
        return out


@ARCH_REGISTRY.register()
class GFPGANv1Clean(nn.Module):
    """The GFPGAN architecture: Unet + StyleGAN2 decoder with SFT.

    It is the clean version without custom compiled CUDA extensions used in StyleGAN2.

    Ref: GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior.

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        decoder_load_path (str): The path to the pre-trained decoder model (usually, the StyleGAN2). Default: None.
        fix_decoder (bool): Whether to fix the decoder. Default: True.

        num_mlp (int): Layer number of MLP style layers. Default: 8.
        input_is_latent (bool): Whether input is latent style. Default: False.
        different_w (bool): Whether to use different latent w for different layers. Default: False.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(
            self,
            out_size,
            num_style_feat=512,
            channel_multiplier=1,
            decoder_load_path=None,
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            input_is_latent=False,
            different_w=False,
            narrow=1,
            sft_half=False):

        super(GFPGANv1Clean, self).__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat

        unet_narrow = narrow * 0.5  # by default, use a half of input channels
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            '32': int(512 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        self.log_size = int(math.log(out_size, 2))
        # 512
        first_out_size = 2**(int(math.log(out_size, 2)))

        self.conv_body_first = nn.Conv2d(3, channels[f'{first_out_size}'], 1)

        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        #   从9到2一共7个
        for i in range(self.log_size, 2, -1):
            #   i = 9, out_channels = 64, in_channels = 32
            #   i = 8, out_channels = 128, in_channels = 64
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode='down'))
            in_channels = out_channels

        self.final_conv = nn.Conv2d(in_channels, channels['4'], 3, 1, 1)

        # upsample
        in_channels = channels['4']
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            self.conv_body_up.append(ResBlock(in_channels, out_channels, mode='up'))
            in_channels = out_channels

        # to RGB
        self.toRGB = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.toRGB.append(nn.Conv2d(channels[f'{2**i}'], 3, 1))

        if different_w:
            linear_out_channel = (int(math.log(out_size, 2)) * 2 - 2) * num_style_feat
        else:
            linear_out_channel = num_style_feat

        #   4096-->8192
        self.final_linear = nn.Linear(channels['4'] * 4 * 4, linear_out_channel)

        # the decoder: stylegan2 generator with SFT modulations
        self.stylegan_decoder = StyleGAN2GeneratorCSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            narrow=narrow,
            sft_half=sft_half)

        # load pre-trained stylegan2 model if necessary
        if decoder_load_path:
            self.stylegan_decoder.load_state_dict(
                torch.load(decoder_load_path, map_location=lambda storage, loc: storage)['params_ema'])
        # fix decoder without updating params
        if fix_decoder:
            for _, param in self.stylegan_decoder.named_parameters():
                param.requires_grad = False

        # for SFT modulations (scale and shift)
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[f'{2**i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            #   i = 3时, out_channels = sft_out_channels = 256
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1), nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_channels, sft_out_channels, 3, 1, 1)))

    def forward(self, x, return_latents=False, return_rgb=True, randomize_noise=True, **kwargs):
        """Forward function for GFPGANv1Clean.

        Args:
            x (Tensor): Input images.
            return_latents (bool): Whether to return style latents. Default: False.
            return_rgb (bool): Whether return intermediate rgb images. Default: True.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
        """
        conditions = []
        unet_skips = []
        out_rgbs = []

        '''
        Degradation Removal U-Net
        '''
        # encoder
        #   x = {Tensor:(1,3,512,512)}--> {Tensor:(1,32,512,512)}
        feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        #   self.log_size = 9
        for i in range(self.log_size - 2):
            #   i = 0, feat = {Tensor:(1,64,256,256)}
            #   i = 1, feat = {Tensor:(1,128,128,128)}
            #   i = 2, feat = {Tensor:(1,256,64,64)}
            #   i = 3, feat = {Tensor:(1,256,32,32)}
            #   7个ResBlock的下采样
            feat = self.conv_body_down[i](feat)
            unet_skips.insert(0, feat)
        #   unet_skips存储了7次conv_body_down之后的feat
        #   feat = {Tensor:(1,256,4,4)}
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)

        # style code
        #   输入为{Tensor:(1,4096)}
        style_code = self.final_linear(feat.view(feat.size(0), -1))
        if self.different_w:
            #   style_code = {Tensor:(1,16,512)}
            #   self.num_style_feat = 512
            style_code = style_code.view(style_code.size(0), -1, self.num_style_feat)

        # decode
        for i in range(self.log_size - 2):
            # add unet skip
            #   feat = {Tensor:(1,256,4,4)}
            #   这里的unet_skips之前把最新得到的feat，即降采样的最小的feat放在最前面，得到一个由小到大的长度为7的feat序列
            #   直接进行简单相加得到的resnet结构
            feat = feat + unet_skips[i]
            # ResUpLayer
            #   7次上采样的ResBlock
            #   i = 0, feat = {Tensor:(1,256,8,8)}
            feat = self.conv_body_up[i](feat)
            # generate scale and shift for SFT layers
            #   i = 0, scale = {Tensor:(1,256,8,8)}
            scale = self.condition_scale[i](feat)
            #   记录scale结果
            conditions.append(scale.clone())
            #   i = 0, shift = {Tensor:(1,256,8,8)}
            shift = self.condition_shift[i](feat)
            conditions.append(shift.clone())
            # generate rgb images
            if return_rgb:
                out_rgbs.append(self.toRGB[i](feat))

        '''
        Pretainded StyleGAN
        '''
        # decoder
        image, _ = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=return_latents,
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=randomize_noise)

        return image, out_rgbs

import math
import random
import torch
from basicsr.archs.stylegan2_arch import (ConvLayer, EqualConv2d, ScaledLeakyReLU, StyleGAN2Generator)
from basicsr.ops.fused_act import FusedLeakyReLU
from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn
from torch.nn import functional as F
from gfpgan.archs.restormer_arch import (OverlapPatchEmbed, TransformerSHWAMBlock, Downsample, Upsample,
                                         TransformerBlockIn)
import numpy as np


class StyleGAN2GeneratorSFT(StyleGAN2Generator):
    """StyleGAN2 Generator with SFT modulation (Spatial Feature Transform).

    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel magnitude. A cross production will be
            applied to extent 1D resample kernel to 2D resample kernel. Default: (1, 3, 3, 1).
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): The narrow ratio for channels. Default: 1.
        sft_half (bool): Whether to apply SFT on half of the input channels. Default: False.
    """

    def __init__(self,
                 out_size,
                 num_style_feat=512,
                 num_mlp=8,
                 channel_multiplier=2,
                 resample_kernel=(1, 3, 3, 1),
                 lr_mlp=0.01,
                 narrow=1,
                 sft_half=False):
        super(StyleGAN2GeneratorSFT, self).__init__(
            out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            resample_kernel=resample_kernel,
            lr_mlp=lr_mlp,
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
        """Forward function for StyleGAN2GeneratorSFT.

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
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        skips = []
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], noise[1::2],
                                                        noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)

            # the conditions may have fewer levels
            if i < len(conditions):
                # SFT part to combine the conditions
                if self.sft_half:  # only apply SFT to half of the channels
                    # out = {Tensor:(1, 512, 8, 8)}
                    # out = {Tensor:(1, 512, 16, 16)}
                    out_same, out_sft = torch.split(out, int(out.size(1) // 2), dim=1)
                    out_sft = out_sft * conditions[i - 1] + conditions[i]
                    out = torch.cat([out_same, out_sft], dim=1)
                else:  # apply SFT to all the channels
                    out = out * conditions[i - 1] + conditions[i]

            out = conv2(out, latent[:, i + 1], noise=noise2)
            # input = {Tensor:(1, 512, 4, 4}
            # i = 1, out = {Tensor:(1, 512, 8, 8}
            # i = 3, out = {Tensor:(1, 512, 16, 16}
            # i = 5, out = {Tensor:(1, 512, 32, 32}
            # i = 7, out = {Tensor:(1, 256, 64, 64)}
            # i = 9, out = {Tensor:(1, 128, 128, 128)}
            # i = 11, out = {Tensor:(1, 64, 256, 256)}
            # i = 13, out = {Tensor:(1, 32, 512, 512)}
            # i = 15
            skip = to_rgb(out, latent[:, i + 2], skip)  # feature back to the rgb space
            skips.append(skip)
            # i = 1, skip = {Tensor:(1, 3, 8, 8)}
            # i = 13, skip = {Tensor:(1, 3, 512, 512)}
            # i = 15, 跳出循环
            i += 2

        image = skip

        if return_latents:
            return image, skips
        else:
            return image, None


class ConvUpLayer(nn.Module):
    """Convolutional upsampling layer. It uses bilinear upsampler + Conv.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input. Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 bias_init_val=0,
                 activate=True):
        super(ConvUpLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.scale is used to scale the convolution weights, which is related to the common initializations.
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

        if bias and not activate:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

        # activation
        if activate:
            if bias:
                self.activation = FusedLeakyReLU(out_channels)
            else:
                self.activation = ScaledLeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, x):
        # bilinear upsample
        out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # conv
        out = F.conv2d(
            out,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        # activation
        if self.activation is not None:
            out = self.activation(out)
        return out


class ResUpBlock(nn.Module):
    """Residual block with upsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """

    def __init__(self, in_channels, out_channels):
        super(ResUpBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvUpLayer(in_channels, out_channels, 3, stride=1, padding=1, bias=True, activate=True)
        self.skip = ConvUpLayer(in_channels, out_channels, 1, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


class ResidualUpBlock(nn.Module):
    """Residual block with upsampling.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualUpBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvUpLayer(in_channels, out_channels, 3, stride=1, padding=1, bias=True, activate=True)
        self.skip = ConvUpLayer(in_channels, out_channels, 1, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


@ARCH_REGISTRY.register()
class RESTORMGANSHWAM(nn.Module):
    def __init__(
            self,
            inp_channels=3,
            out_channels=3,
            dim=16,
            num_blocks=[2, 3, 3, 4],
            num_refinement_blocks=2,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',  ## Other option 'BiasFree'
            dual_pixel_task=False,  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
            out_size=512,
            num_style_feat=512,
            channel_multiplier=1,
            resample_kernel=(1, 3, 3, 1),
            decoder_load_path='experiments/pretrained_models/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth',
            fix_decoder=True,
            # for stylegan decoder
            num_mlp=8,
            lr_mlp=0.01,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True):

        super(RESTORMGANSHWAM, self).__init__()
        self.input_is_latent = input_is_latent
        self.different_w = different_w
        self.num_style_feat = num_style_feat
        self.style_code = np.load('experiments/pretrained_models/mean_latents.npy')
        self.style_code = torch.Tensor(self.style_code).cuda()

        # dim = 16
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerSHWAMBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                  LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerSHWAMBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerSHWAMBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerSHWAMBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerSHWAMBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerSHWAMBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerSHWAMBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type, flag=True) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerSHWAMBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                  bias=bias, LayerNorm_type=LayerNorm_type, flag=True) for i in
            range(num_refinement_blocks)])

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        ###########################

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        unet_narrow = narrow * 0.5  # by default, use a half of input channels
        # channels = {'4': 256, '8': 256, '16': 256, '32': 256, '64': 128, '128': 64, '256': 32, '512': 16, '1024': 8}
        channels = {
            '4': int(512 * unet_narrow),
            '8': int(512 * unet_narrow),
            '16': int(512 * unet_narrow),
            # '32': int(512 * unet_narrow),
            '32': int(256 * unet_narrow),
            '64': int(256 * channel_multiplier * unet_narrow),
            '128': int(128 * channel_multiplier * unet_narrow),
            '256': int(64 * channel_multiplier * unet_narrow),
            '512': int(32 * channel_multiplier * unet_narrow),
            '1024': int(16 * channel_multiplier * unet_narrow)
        }

        self.log_size = int(math.log(out_size, 2))

        # to RGB
        # self.toRGB = nn.ModuleList()
        # for i in range(6, self.log_size + 1):
        #     self.toRGB.append(EqualConv2d(channels[f'{2**i}'], 3, 1, stride=1, padding=0, bias=True, bias_init_val=0))

        # the decoder: stylegan2 generator with SFT modulations
        self.stylegan_decoder = StyleGAN2GeneratorSFT(
            out_size=out_size,
            num_style_feat=num_style_feat,
            num_mlp=num_mlp,
            channel_multiplier=channel_multiplier,
            resample_kernel=resample_kernel,
            lr_mlp=lr_mlp,
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
        for i in range(3, 6):
            out_channels = channels[f'{2 ** i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2
            if i == 5:
                self.condition_scale.append(ConditionSample(out_channels, sft_out_channels * 2, mode='double_chann'))
                self.condition_shift.append(ConditionSample(out_channels, sft_out_channels * 2, mode='double_chann'))
            else:
                self.condition_scale.append(ConditionSample(out_channels, sft_out_channels))
                self.condition_shift.append(ConditionSample(out_channels, sft_out_channels))

        for i in range(6, self.log_size + 1):
            out_channels = channels[f'{2 ** i}']
            if sft_half:
                sft_out_channels = out_channels
            else:
                sft_out_channels = out_channels * 2

            if i == 9:
                out_channels = 32
                sft_out_channels = 16

            self.condition_scale.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=1)))
            self.condition_shift.append(
                nn.Sequential(
                    EqualConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0),
                    ScaledLeakyReLU(0.2),
                    EqualConv2d(out_channels, sft_out_channels, 3, stride=1, padding=1, bias=True, bias_init_val=0)))

    def forward(self, x, return_latents=False, return_rgb=True, randomize_noise=True, **kwargs):
        """Forward function for GFPGANv1.

        Args:
            x (Tensor): Input images.
            return_latents (bool): Whether to return style latents. Default: False.
            return_rgb (bool): Whether return intermediate rgb images. Default: True.
            randomize_noise (bool): Randomize noise, used when 'noise' is False. Default: True.
        """
        conditions = []
        out_rgbs = []

        # encoder
        inp_enc_level1 = self.patch_embed(x)
        # 残差特征512
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        # 残差特征256
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        # 残差特征128
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        # 残差特征64
        latent = self.latent(inp_enc_level4)
        # condition
        scale = self.condition_scale[3](latent)
        conditions.append(scale.clone())
        shift = self.condition_shift[3](latent)
        conditions.append(shift.clone())

        for i in range(3):
            scale = self.condition_scale[2 - i](scale)
            conditions.insert(0, scale.clone())
            shift = self.condition_shift[2 - i](shift)
            conditions.insert(0, shift.clone())

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        # condition
        scale = self.condition_scale[4](out_dec_level3)
        conditions.append(scale.clone())
        shift = self.condition_shift[4](out_dec_level3)
        conditions.append(shift.clone())

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        # condition
        scale = self.condition_scale[5](out_dec_level2)
        conditions.append(scale.clone())
        shift = self.condition_shift[5](out_dec_level2)
        conditions.append(shift.clone())

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        # condition
        scale = self.condition_scale[6](out_dec_level1)
        conditions.append(scale.clone())
        shift = self.condition_shift[6](out_dec_level1)
        conditions.append(shift.clone())

        # For Dual-Pixel Defocus Deblurring Task #
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + x

        if return_rgb:
            out_rgbs.append(out_dec_level1)

        style_code = self.style_code.repeat(x.size(0), 1, 1)

        # decoder
        image, _ = self.stylegan_decoder([style_code],
                                         conditions,
                                         return_latents=return_latents,
                                         input_is_latent=self.input_is_latent,
                                         randomize_noise=randomize_noise)

        return image, out_rgbs


class ConditionSample(nn.Module):
    def __init__(self, in_channels, out_channels, mode=None):
        super(ConditionSample, self).__init__()

        if mode == 'double_chann':
            self.downsample = Downsample(in_channels)
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.downsample = Downsample2(in_channels)
            self.skip = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.scale_factor = 0.5

    def forward(self, x):
        out = self.downsample(x)
        # skip
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        skip = self.skip(x)
        out = out + skip
        return out


class Downsample2(nn.Module):
    def __init__(self, n_feat):
        super(Downsample2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

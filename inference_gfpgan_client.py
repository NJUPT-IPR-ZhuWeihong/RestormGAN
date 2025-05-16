import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from basicsr.utils.options import yaml_load

from gfpgan.utils_client import GFPGANer2


def main():
    """Inference demo for GFPGAN (for users).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default='inputs/whole_imgs',
        help='Input image or folder. Default: inputs/whole_imgs')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
    # we use version to select models, which is more user-friendly
    parser.add_argument(
        '-v', '--version', type=str, default='1.3', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    parser.add_argument(
        '-s', '--upscale', type=int, default=2, help='The final upsampling scale of the image. Default: 2')
    parser.add_argument(
        '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument(
        '--bg_tile',
        type=int,
        default=0,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)

    # ------------------------ set up background upsampler ------------------------
    upsampler_model_path = 'upsampler_weights/RealESRGAN_x2plus.pth'
    if args.bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                          'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            #   ESRGAN网络的基本单元Residual-in-Residual Dense Block (RRDB)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            #   在GFPGAN网络中作为上采样器
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path=upsampler_model_path,
                model=model,
                tile=args.bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    with_mask = False
    aligned = False
    if args.version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif args.version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif args.version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif args.version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif args.version == 'net_g':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'net_g_latest'
    elif args.version == 'net_g_2':
        arch = 'original'
        channel_multiplier = 2
        model_name = 'net_g_500000'
    elif args.version == 'test':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'test2'
    elif args.version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
    elif args.version == 'RestormGAN':
        arch = 'restorm'
        channel_multiplier = 1
        model_name = 'RestormGAN_degre'
    elif args.version == 'RestormGAN_degre':
        arch = 'restorm'
        aligned = True
        channel_multiplier = 1
        model_name = 'RestormGAN_StyleFFHQ_degre'
    elif args.version == 'URestormGAN':
        arch = 'urestorm'
        channel_multiplier = 1
        model_name = 'with_space_attent'
    elif args.version == 'RestormGAN_tiny':
        arch = 'restorm'
        channel_multiplier = 1
        model_name = 'restormgan_tiny'
    elif args.version == 'RestormFaceGAN':
        arch = 'restormface'
        channel_multiplier = 1
        model_name = 'restormfaceganv1'
        with_mask = True
    elif args.version == 'RestormGAN_half':
        arch = 'restormhalf'
        channel_multiplier = 1
        model_name = 'restormgan_half'
    elif args.version == 'RestormGAN_CNN':
        arch = 'restormCNN'
        channel_multiplier = 1
        model_name = 'RestormGAN_CNN3'
    elif args.version == 'GFPGAN_UBlock':
        arch = 'GFPGAN_u'
        channel_multiplier = 1
        model_name = 'GFPGAN_Ublock'
    elif args.version == 'GFPGAN_UBlock_CFF':
        arch = 'GFPGAN_u_cff'
        aligned = True
        channel_multiplier = 2
        model_name = 'GFPGAN_UBlock_CFF'
    elif args.version == 'GFPGAN_UBlock_latents':
        arch = 'GFPGAN_u_latents'
        aligned = True
        channel_multiplier = 2
        model_name = 'GFPGAN_Ublock_latentschann2'
    elif args.version == 'GFPGAN_resblock_chann2':
        arch = 'GFPGAN_res_2'
        aligned = True
        channel_multiplier = 2
        model_name = 'GFPGAN_resblock_FFHQchann2'
    elif args.version == 'GFPGAN_CFFchann1':
        arch = 'original'
        aligned = True
        channel_multiplier = 1
        model_name = 'GFPGAN_CFFchann1_10w'
    elif args.version == 'GFPGAN_resblock':
        arch = 'GFPGAN_res'
        channel_multiplier = 1
        model_name = 'GFPGAN_resblock'
    elif args.version == 'GFPGAN_FFHQdegre':
        arch = 'original'
        aligned = True
        channel_multiplier = 1
        model_name = 'GFPGAN_FFHQdegre2'
    elif args.version == 'UformerGAN':
        arch = 'Uformer'
        channel_multiplier = 1
        model_name = 'UformerGAN'
    elif args.version == 'refine':
        arch = 'refine'
        channel_multiplier = 1
        model_name = 'GFPGAN_refine'
    elif args.version == 'GFPGAN_resblock2':
        arch = 'resblock2'
        channel_multiplier = 1
        model_name = 'GFPGAN_resblock2_CelebA'
    elif args.version == 'RestormGAN_FFHQdegre':
        arch = 'RestormGAN1223'
        channel_multiplier = 1
        model_name = 'RestormGAN_FFHQdegre'
    elif args.version == 'RestormGAN_FFHQdegre_crop':
        arch = 'RestormGAN1223'
        aligned = True
        channel_multiplier = 1
        model_name = 'RestormGAN_manualCG_crop'
    elif args.version == 'RestormGAN_CBAM_FFHQdegre':
        arch = 'RestormGAN1223_cbam'
        aligned = True
        channel_multiplier = 1
        model_name = 'RestormGAN_CBAM_FFHQdegre'
    else:
        raise ValueError(f'Wrong model version {args.version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    opt = yaml_load(args.opt)

    restorer = GFPGANer2(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler,
        opt=opt['datasets']['train'])

    # ------------------------ restore ------------------------
    count = 0
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        '''
        采用GFP-GAN生成的图片无论如何都是Tensor(1, 3, 512, 512)形状大小的
        若输入图片为(3, 128, 128)，则输出为(3, 256, 256)的图片会被置于512*512图片的正中间
        若想得到中间所需大小的图片，则需要对输出进行仿射变换
        '''
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            img_path,
            has_aligned=aligned,
            only_center_face=args.only_center_face,
            paste_back=True,
            weight=args.weight,
            count=count,
            with_mask=with_mask)

        count += 1

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            # save cropped face
            save_crop_path = os.path.join(args.output, 'cropped_faces', f'{basename}_{idx:02d}.png')
            imwrite(cropped_face, save_crop_path)
            # save restored face
            if args.suffix is not None:
                save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(args.output, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            if args.ext == 'auto':
                extension = ext[1:]
            else:
                extension = args.ext

            if args.suffix is not None:
                save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
            else:
                save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)

    print(f'Results are in the [{args.output}] folder.')


if __name__ == '__main__':
    main()

import argparse
import glob
import os
import cv2
from gfpgan.utils import GFPGANer
from gfpgan.data.ffhq_degradation import FFHQDegradation
from basicsr.utils.options import yaml_load
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from torchvision import utils


def main():
    """Inference demo for GFPGAN (for users).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to option YAML file.')
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
        '--bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
    parser.add_argument('-f', '--flag', type=bool, default=False, help='flag to decide whether to degrade images')
    args = parser.parse_args()
    opt = yaml_load(args.opt)

    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)

    # ------------------------ set up GFPGAN restorer ------------------------
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
    else:
        raise ValueError(f'Wrong model version {args.version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier)

    degradater = FFHQDegradation(opt['datasets']['train'])

    # ------------------------ restore ------------------------
    latents_list = []
    img_count = 0
    for img_path in img_list:
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        # if input is high-resolution
        if args.flag:
            img_lr = degradater.img_deg(img_path)
        # if input is low-resolution
        else:
            img_lr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_lr = cv2.resize(img_lr, (512, 512))
            img_lr = img2tensor(img_lr / 255., bgr2rgb=True, float32=True)
            normalize(img_lr, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)

        # restore faces and background if necessary
        '''
        采用GFP-GAN生成的图片无论如何都是Tensor(1, 3, 512, 512)形状大小的
        若输入图片为(3, 128, 128)，则输出为(3, 256, 256)的图片会被置于512*512图片的正中间
        若想得到中间所需大小的图片，则需要对输出进行仿射变换
        '''
        # latents, ulti_img, sr_latents, inter_img = restorer.infer_latents(img_lr)
        ulti_img, inter_img = restorer.infer_latents(img_lr)
        # latents_np = latents[0].cpu().detach().numpy()
        # latents_list.append(latents_np)
        utils.save_image(
            img_lr,
            f'datasets/visual_severe_deg/lr/{str(img_count).zfill(6)}.png',
            normalize=True,
            range=(-1, 1),
        )
        utils.save_image(
            ulti_img,
            f'datasets/visual_severe_deg/sr/{str(img_count).zfill(6)}.png',
            normalize=True,
            range=(-1, 1),
        )
        for i in range(len(inter_img)):
            utils.save_image(
                inter_img[i],
                f'datasets/visual_severe_deg/skips{str(i)}/{str(img_count).zfill(6)}.png',
                normalize=True,
                range=(-1, 1),
            )
        img_count += 1

    # save latents
    # np.save('datasets/latents/latents_lr/latents_lr.npy', latents_list)

    print(f'Results are in the [{args.output}] folder.')


if __name__ == '__main__':
    main()

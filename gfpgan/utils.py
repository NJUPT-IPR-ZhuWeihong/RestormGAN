import cv2
import os
import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from torch.nn import functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class GFPGANer():
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(self, model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=None):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        # initialize the GFP-GAN
        if arch == 'clean':
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'bilinear':
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'original':
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True)
        elif arch == 'RestoreFormer':
            from gfpgan.archs.restoreformer_arch import RestoreFormer
            self.gfpgan = RestoreFormer()

        elif arch == 'shwam2334':
            from gfpgan.archs.restormgan_shwam_arch import RESTORMGANSHWAM
            self.gfpgan = RESTORMGANSHWAM(num_blocks=[2, 3, 3, 4])

        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            device=self.device,
            model_rootpath='gfpgan/weights')

        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=os.path.join(ROOT_DIR, 'gfpgan/weights'), progress=True, file_name=None)
        loadnet = torch.load(model_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5, count=None,
                with_mask=False):
        self.face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            #   将16-bit图像、grey图像、RGBA图像转化为正常RGB图像
            self.face_helper.read_image(img)
            '''
            （1）这里会自动加载一个人脸检测模型，比方retinaface对图片进行人脸检测，对检测到的人脸提取位置信息
            （2）调用retinaface检测到的bboxes，先检查两只眼睛的大小，若小于一定值则去除该人脸
            （3）landmarks_5指的是bboxes列表中的5，7，9，11，13个数
            （4）det_faces指的是det_faces中的0，1，2，3，4
            （5）从self.det_faces得到图片最中心的人脸
            （6）根据最中心人脸的索引得到对应的landmarks_5
            （7）利用landmarks_5信息得到最佳仿射矩阵从而对人脸图片进行变换
            '''
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            # crop_face的同时会根据scale值 128-->512，多余的部分pad填充
            self.face_helper.align_warp_face()
            #   得到cropped_face文件夹

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            #   将图片转化为张量
            #   {ndarray:(512,512,3)}-->{Tensor:(3,512,512)}
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            #   cropped_face_t = {Tensor:(1,3,512,512)}
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            if with_mask:
                self.get_infer_mask(input=cropped_face_t, labelpath="D:/FSR_Code/RestormGAN/inputs/annotations/",
                                    count=count)

            #   此时经过预处理过的人脸再放进GFPGAN网络中
            try:
                #   cropped_face_t = output = {Tensor:(1,3,512,512)}
                if with_mask:
                    output = self.gfpgan(cropped_face_t, return_rgb=False, weight=weight, mask=self.face_arrays)[0]
                else:
                    output = self.gfpgan(cropped_face_t, return_rgb=False, weight=weight)[0]
                # convert to image
                #   restored_face = {Tensor:(512,512,3)}
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f'\tFailed inference for GFPGAN: {error}.')
                restored_face = cropped_face

            # Tensor:(1, 3, 512, 512)
            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None

    @torch.no_grad()
    def infer_latents(self, img):
        img = img.unsqueeze(0).to(self.device)
        # latents, ulti_img, sr_latents, inter_img = self.gfpgan(img, return_latents=True, return_rgb=True)
        ulti_img, inter_img = self.gfpgan(img, return_latents=True, return_rgb=False)
        # return latents, ulti_img, sr_latents, inter_img
        return ulti_img, inter_img

    def get_infer_mask(self, input, labelpath, count):
        array = input
        filelist = os.listdir(labelpath)
        fileIndex = []
        self.face_arrays = []

        # 文件名读入时并非按照我们常识中的按照文件名字顺序读入，
        # 例如：1.json,2.json,3.json；程序可能会按 3,1,2 的顺序读入，
        # 这对我们后面批量处理造成很大的不便，所以读入文件名后，
        # 我们要手动地对文件名进行一次排序
        # 以下就是排序操作
        for i in range(0, len(filelist)):
            index = filelist[i].split(".")[0]
            fileIndex.append(int(index))
        # new_filelist =[]
        for j in range(1, len(fileIndex)):
            for k in range(0, len(fileIndex) - 1):
                if fileIndex[k] > fileIndex[k + 1]:
                    preIndex = fileIndex[k]
                    preFile = filelist[k]
                    fileIndex[k] = fileIndex[k + 1]
                    filelist[k] = filelist[k + 1]
                    fileIndex[k + 1] = preIndex
                    filelist[k + 1] = preFile

        # 完成排序后，开始按照文件名顺序读取文件内容信息
        face_arrays = []
        with open(labelpath + filelist[count], 'r') as txt:
            lines = txt.readlines()
            for each in range(0, len(lines)):
                word = lines[each].split('"')
                for i in range(0, len(word)):
                    if word[i] == 'lefteye':
                        xmin = int(float(lines[each + 3].split(',')[0]))
                        ymin = int(float(lines[each + 4].split('\n')[0]))
                        xmax = int(float(lines[each + 7].split(',')[0]))
                        ymax = int(float(lines[each + 8].split('\n')[0]))
                        # array[:, :, xmin:xmax, ymin:ymax] = 1
                        # (xmin,ymin,xmax,ymax,'left_eye')
                    elif word[i] == 'righteye':
                        xmin = int(float(lines[each + 3].split(',')[0]))
                        ymin = int(float(lines[each + 4].split('\n')[0]))
                        xmax = int(float(lines[each + 7].split(',')[0]))
                        ymax = int(float(lines[each + 8].split('\n')[0]))
                        # array[:, :, xmin:xmax, ymin:ymax] = 1
                        # (xmin,ymin,xmax,ymax,'right_eye')
                    elif word[i] == 'mouth':
                        xmin = int(float(lines[each + 3].split(',')[0]))
                        ymin = int(float(lines[each + 4].split('\n')[0]))
                        xmax = int(float(lines[each + 7].split(',')[0]))
                        ymax = int(float(lines[each + 8].split('\n')[0]))
                        # array[:, :, xmin:xmax, ymin:ymax] = 1
                        # (xmin,ymin,xmax,ymax,'mouth')
            array = array[:, 0, :, :].unsqueeze(1).repeat(1, 32, 1, 1)
            array0 = F.interpolate(array, scale_factor=0.5, mode='bilinear', align_corners=False)
            array0 = array0[:, 0, :, :].unsqueeze(1).repeat(1, 32, 1, 1)
            array1 = F.interpolate(array0, scale_factor=0.5, mode='bilinear', align_corners=False)
            array1 = array1[:, 0, :, :].unsqueeze(1).repeat(1, 64, 1, 1)
            face_arrays.append(array1)
            face_arrays.append(array0)
            face_arrays.append(array)
            self.face_arrays = face_arrays
            txt.close()

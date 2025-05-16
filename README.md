# RestormGAN: Restormer with Generative Facial Prior Towards Real-World Blind Face Restoration
## Introduction
We propose RestormGAN that consists of a CAR encoder and a StyleGAN2 generator for real-world blind face restoration task (i.e., TMDF images) via degradation simulation.
Extensive experiments indicate that our RestormGAN achieves state-of-the-art performances for blind face restoration. Our two-stage trained RestormGAN-S1 and RestormGAN-S2 can efficiently enhance real-world blind face images 
(i.e., TMDF images).

## Datasets
1. [FFHQ](https://github.com/NVlabs/ffhq-dataset) is a publicly available dataset.
2. [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a publicly available dataset.
3. [VGGFace2](https://aistudio.baidu.com/datasetdetail/107435) is a publicly available dataset.
4. LFW is a publicly available dataset.
5. CFF is a private dataset and cannot be made public.
6. TMDF is also a private dataset and cannot be made public.

## Training
We provide the training codes for RestormGAN based on [GFPGAN](https://github.com/TencentARC/GFPGAN?tab=readme-ov-file) (used in our paper).
You could improve it according to your own needs.
```
./gfpgan/train.py
```

### Configuration
Modify the related parameters (paths, loss weights, training steps, and etc.) in the config yaml files.Please refer to [GFPGAN](https://github.com/TencentARC/GFPGAN?tab=readme-ov-file) for relevant settings.
```
./options/train_restormgan.yml
```

## Testing
### Pre-trained models
Please download our pre-trained models via the following links [Baiduyun (extracted code: 1a2b)](https://pan.baidu.com/s/1j7TC79W4S5m4GC5IyiciKA?pwd=1a2b) 
[Google Drive](https://drive.google.com/drive/folders/1leBqBpAZ2QQ432oMihETGFqWwzwloZfl). 
Place the downloaded pre-trained model in the following path。
```
./experiments/pretrained_models
```

```
./inference_gfpgan.py
```

## Citation
If you find this work useful for your research, please cite our paper
```
@article{hu2025restormgan,
  title={RestormGAN: Restormer with generative facial prior towards real-world blind face restoration},
  author={Hu, Changhui and Zhu, Weihong and Xu, Lintao and Wu, Fei and Cai, Ziyun and Ye, Mengjun and Lu, Xiaobo},
  journal={Computers and Electrical Engineering},
  volume={123},
  pages={110095},
  year={2025},
  publisher={Elsevier}
}
```

## Contact
If you have any questions, please feel free to contact the authors via 994628118@qq.com.
    

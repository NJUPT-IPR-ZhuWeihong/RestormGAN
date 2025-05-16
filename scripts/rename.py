import os
import PIL.Image as Image
from tqdm import tqdm

infile = r"results/RestormGAN_perploss=10/restored_faces"
file_dir = os.listdir(infile)
count = 0
for filename in tqdm(file_dir):
    im = Image.open(os.path.join(infile, filename))
    pre_name = os.path.splitext(filename)[0]
    # pre_name = int(pre_name) + 18000
    pre_name = pre_name.split("_")[0]
    # save_path = f'D:/FSR_Code/RestormGAN/results/restored_faces_rename/{str(pre_name).zfill(6)}.png'
    save_path = f'results/RestormGAN_perploss=10/rename/{str(pre_name)}.png'
    im.save(save_path)
    count += 1
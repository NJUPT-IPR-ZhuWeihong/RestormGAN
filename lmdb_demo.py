from basicsr.utils.lmdb_util import make_lmdb_from_imgs
import os
from tqdm import tqdm
data_path = r"datasets/ffhq/ffhq_256"
lmdb_path = r"datasets/ffhq/ffhq_256.lmdb"
img_list = os.listdir(data_path)
keys_init = range(1000)
keys = []
for key in keys_init:
    keys.append(str(key))
for img in range(len(img_list)):
    abs_path = os.path.join(data_path, img_list[0])
    make_lmdb_from_imgs(data_path, lmdb_path, img_list, keys)

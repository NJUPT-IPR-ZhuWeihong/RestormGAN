import cv2
import numpy as np
from shutil import copyfile, move
import os
import time



def get_file_list(path, suffix):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(suffix):
                Filelist.append(os.path.join(home, filename))
    return Filelist


if __name__ == "__main__":
    '''
        实现对test目录下图片进行分类，并将结果保存在results下
    '''
    # src_path = r"D:\src"
    src_path = r"E:\DATASET\crop_face_dataset\Traffic_checkpoint\face"
    dst_path = r"E:\DATASET\crop_face_dataset\Traffic_checkpoint\scale_classification"
    sizefilter = [[30, 30], [40, 40], [50, 50], [60, 60], [70, 70], [700, 1400], [1000, 1700],
                  [2100, 1800], [2000, 1400]]
    foldername = ["size_30_30", "size_40_40", "other", "size_50_50", "size_60_60", "size_6_15", "size_7_14", "size_10_17",
                  "size_21_18", "size_20_14"]
    Filelist = get_file_list(src_path, ".png")
    print(len(Filelist))
    for filename in Filelist:
        sz = os.path.getsize(filename)
        if sz == 0:
            continue
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), 1)

        if img is None:
            continue
        (H, W) = img.shape[:2]
        # 将图片按不同尺寸分类保存
        midname = dst_path
        if 900 <= (H * W) <= 1600:
            midname = os.path.join(midname, foldername[0])
        elif 1600 < (H * W) <= 2500:
            midname = os.path.join(midname, foldername[1])
        elif 2500 < (H * W) <= 3600:
            midname = os.path.join(midname, foldername[3])
        elif 3600 < (H * W):
            midname = os.path.join(midname, foldername[4])
        else:
            midname = os.path.join(midname, foldername[2])
        # if sizefilter[0][0] <= H <= sizefilter[0][0] + 20 and sizefilter[0][1] <= W < sizefilter[0][1] + 10:
        #     midname = os.path.join(midname, foldername[0])
        # elif sizefilter[1][0] <= H <= sizefilter[1][0] + 10 and sizefilter[1][1] <= W < sizefilter[1][1] + 10:
        #     midname = os.path.join(midname, foldername[1])
        # elif sizefilter[2][0] <= H <= sizefilter[2][0] + 10 and sizefilter[2][1] <= W < sizefilter[2][1] + 10:
        #     midname = os.path.join(midname, foldername[3])
        # elif sizefilter[3][0] <= H <= sizefilter[3][0] + 10 and sizefilter[3][1] <= W < sizefilter[3][1] + 10:
        #      midname = os.path.join(midname, foldername[4])
        # elif sizefilter[4][0] < H < sizefilter[4][0] + 100 and sizefilter[4][1] < W < sizefilter[4][1] + 100:
        #     midname = os.path.join(midname, foldername[5])
        # elif sizefilter[5][0] < H < sizefilter[5][0] + 100 and sizefilter[5][1] < W < sizefilter[5][1] + 100:
        #     midname = os.path.join(midname, foldername[6])
        # elif sizefilter[6][0] < H < sizefilter[6][0] + 100 and sizefilter[6][1] < W < sizefilter[6][1] + 100:
        #     midname = os.path.join(midname, foldername[7])
        # elif sizefilter[7][0] < H < sizefilter[7][0] + 200 and sizefilter[7][1] < W < sizefilter[7][1] + 200:
        #     midname = os.path.join(midname, foldername[8])
        # elif sizefilter[8][0] < H < sizefilter[8][0] + 200 and sizefilter[8][1] < W < sizefilter[8][1] + 200:
        #     midname = os.path.join(midname, foldername[9])
        # else:
        #      midname = os.path.join(midname, foldername[2])

        file = filename.rsplit('\\', 1)[1]
        output_name_file = os.path.join(midname, file)
        output_name_dir = output_name_file.rsplit('\\', 1)[0]
        if not os.path.exists(output_name_dir):
            os.makedirs(output_name_dir)
        if filename == output_name_file:
            continue
        copyfile(filename, output_name_file)
        # move(filename, output_name_file)




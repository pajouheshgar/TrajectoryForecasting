import os
basedir = "Dataset/SDD/annotations/"
folders = os.listdir(basedir)

for f in folders:
    videos = os.listdir(basedir + f + "/")
    for v in videos:
        os.remove(basedir + f + "/" + v + "/frame_dict.npy")
        os.remove(basedir + f + "/" + v + "/object_dict.npy")

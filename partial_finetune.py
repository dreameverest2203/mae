import os, random, glob
import pdb
import shutil

source_train = "/data5/chengxuz/Dataset/imagenet_raw/train/"
dest_train = "/data2/aamdekar/ImageNet_10/train"

for img_class in os.listdir(source_train):
    class_id = os.path.join(source_train, img_class).split("/")[-1]
    new_dir = os.path.join(source_train, class_id)
    dest_dir = os.path.join(dest_train, class_id)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    ls_img = glob.glob(os.path.join(new_dir, "*.JPEG"))
    for file_name in random.sample(ls_img, 100):
        shutil.copy(file_name, dest_dir)

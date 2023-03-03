import os
from sklearn.model_selection import KFold
import numpy as np
import shutil

kf = KFold(n_splits=5,shuffle=True,random_state=42)

root_path = "./TsinghuaDog/angry"
dst_path = "./FiveFold"
index = 0
img_path = np.array(os.listdir(root_path))
for train_index,test_index in kf.split(img_path):
    # print("Train Index:", train_index, ",Test Index:", test_index)
    dst = os.path.join(dst_path,str(index))
    X_train, X_test = img_path[train_index], img_path[test_index]
    for i in X_train:
        ori_path = os.path.join(root_path,i)
        image_path = os.path.join(dst,"train","angry",i)
        shutil.copyfile(ori_path,image_path)
    for i in X_test:
        ori_path = os.path.join(root_path, i)
        image_path = os.path.join(dst, "val", "angry",i)
        shutil.copyfile(ori_path, image_path)
    index = index + 1
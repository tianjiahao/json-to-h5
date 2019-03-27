# 合并两个h5文件 Date: 2018.12.05
import h5py
import numpy as np

#读取h5信息
f = h5py.File('/home/myubuntu/Desktop/姿态估计标注0张-30张/1000/alpha_train2017-1000_new.h5', 'r')
for k in f.keys():
    print(k)
bnds1 = f['bndbox'].value
imgs1 = f['imgname'].value
parts1 = f['part'].value

f = h5py.File('/home/myubuntu/Desktop/姿态估计标注0张-30张/alpha_val2017.h5', 'r')
for k in f.keys():
    print(k)
bnds2 = f['bndbox'].value
imgs2 = f['imgname'].value
parts2 = f['part'].value

#融合两个h5文件的imgname
imgs = np.append(imgs1, imgs2)
imgs=imgs.tolist()
#print(imgs)
h5_imgs=[]
for k in range(len(imgs)):
    if (k % 16 == 0):
        tep2 = []
    tep2.append(imgs[k])
    if (k % 16 == 15):
        h5_imgs.append(tep2)
h5_imgs = np.array(h5_imgs)
#print(h5_imgs)

#融合两个h5文件的boundingbox
bnds = np.append(bnds1, bnds2)
bnds = bnds.tolist()
# print(bnds)
h5_bnds = []
for k in range(len(bnds)):
    if (k % 4 == 0):
        tmp1 = []
    tmp1.append(bnds[k])
    if (k % 4 != 3):
        tmp2 = []
        tmp2.append(tmp1)
    if (k % 4 == 3):
        h5_bnds.append(tmp2)
h5_bnds = np.array(h5_bnds)
#print(h5_bnds)

#融合两个h5文件的keypoint
parts = np.append(parts1, parts2)
parts = parts.astype(np.int)
#print(parts)
parts = parts.tolist()
h5_parts = []
tkp1=[]
for k in range(len(parts)):
    if (k % 2 == 0):
        tkp = []
    tkp.append(parts[k])
    if (k % 2 == 1):
        tkp1.append(tkp)
#tkp1 = np.array(tkp1)
#print(tkp1)
for i in range(len(tkp1)):
    if (i % 17 == 0):
        tkp2 = []
    tkp2.append(tkp1[i])
    if (i % 17 == 16):
        h5_parts.append(tkp2)
h5_parts = np.array(h5_parts)
#print(h5_parts)

#写入h5文件
h5file = h5py.File('/home/myubuntu/Desktop/姿态估计标注0张-30张/1000/alpha_train2017-1000_new_alpha_val2017.h5','w')
h5file.create_dataset('imgname', data=h5_imgs)
h5file.create_dataset('bndbox',data=h5_bnds)
h5file.create_dataset('part',data=h5_parts)


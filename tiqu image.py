import os
import numpy as np
import shutil
import time


time1= time.time()
filename = "/home/myubuntu/bad_name.txt"
rootdir = "/home/myubuntu/Music/new-AlphaPose-pytorch/examples/ibn_m_val2017/vis"
dstdir = "/home/myubuntu/bad_name"
imagename=[]
for line in open(filename,"r"):
    imagename.append(line.split('.')[0])
lst = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
for i in range(0,len(lst)):
    path = os.path.join(rootdir,lst[i])
    path1=path[-16: -4]
    for j in range(0,len(imagename)):
        #imagename=imagename[j]
        #print(imagename)
        if imagename[j] == path1:
            shutil.copy(path, dstdir)
time2=time.time()
print('spent time = %.2f min'%((time2-time1)/60))
print('well done!')

import h5py

f = h5py.File('/home/myubuntu/Desktop/姿态估计标注0张-30张/1000/alpha_train2017-1000_new_alpha_val2017.h5','r')
for k in f.keys():
    print(k)
b=f['bndbox']
i=f['imgname']
p=f['part']
print(b.shape)
print(i.shape)
print(p.shape)


import h5py
with h5py.File('/home/myubuntu/Desktop/python_files/h5/annot_coco.h5', 'r') as annot:
    # train
    imgname_coco_train = annot['imgname'][:5000]
    bndbox_coco_train = annot['bndbox'][:5000]
    part_coco_train = annot['part'][:5000]
#写入h5文件
h5file = h5py.File('/home/myubuntu/Desktop/姿态估计标注0张-30张/alpha_train2017_5000.h5','w')
h5file.create_dataset('imgname', data=imgname_coco_train)
h5file.create_dataset('bndbox',data=bndbox_coco_train)
h5file.create_dataset('part',data=part_coco_train)


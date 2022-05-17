import os
import PIL.Image as Image

rootdir = '/home/newdisk/zekai.shao/pytorch-CycleGAN-and-pix2pix/results/basic_cyclegan_default_para/test_latest/'

dir1 = os.path.join(rootdir, 'set1')
dir2 = os.path.join(rootdir, 'set2')

if not os.path.exists(dir1):
    os.makedirs(dir1)
    os.makedirs(dir2)

for file in os.listdir(rootdir+'images/'):
    img = Image.open(os.path.join(rootdir, 'images', file))
    flag = file.split('_')[1].split('.')[0]
    if flag == 'real':
        img.save(os.path.join(dir1, file))
    elif flag == 'fake':
        img.save(os.path.join(dir2, file))
    else:
        raise NotImplementedError
import cv2
import os
import random 
import mindspore
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.dataset as ds
from mindspore.dataset.vision.c_transforms import Inter,RandomRotation,Resize

def get_files(path):
    ret = []
    for tuple_path in os.walk(path):
        for filespath in tuple_path[2]:
            ret.append(os.path.join(tuple_path[0], filespath))
    return ret,len(ret)
   
def read_masks(file):
    """读取图片，图片格式为hwc,返回numpy"""
    img = cv2.imread(file)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def random_rotate_image(image):
    rotate = RandomRotation(90,Inter.NEAREST)
    return rotate(image)

def random_resize_image(image, scale, height, width):
    newsize = [int(height*scale), int(width*scale)]
    resize = Resize(newsize,Inter.NEAREST)
    return resize(image)

def random_mask(args):
    img_shape = args.IMG_SHAPE
    height = img_shape[0]
    width = img_shape[1]
    path_list,n_masks = get_files(args.mask_template_dir)
    nd = random.randint(0,n_masks-1)
    path_mask = path_list[nd]
    mask = read_masks(path_mask)#256,256,3
    mask = ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5)(mask)
    scale = random.uniform(0.8,1.0)
    mask = random_rotate_image(mask)
    mask = random_resize_image(mask,scale,height,width) #506,506,3
    crop = ds.vision.c_transforms.CenterCrop((height,width))
    mask1 = crop(mask)
    mask2 = Tensor.from_numpy(mask1)
    mask3 = mask2.astype(mindspore.float32)
    mask4 = mask3[:,:,0:1]
    mask5 = ops.ExpandDims()(mask4,0) #1,512,512,1
    mask6 = ops.Mul()(1/255,mask5)
    mask = ops.Reshape()(mask6,(1,height, width,1))  #NHWC
    mask = ops.Transpose()(mask,(0,3,1,2))
    return mask

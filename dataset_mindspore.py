import os
import cv2
import math
import numpy as np

def np_scale_to_shape(image, shape):
    """
    Scale the image.
    The minimum side of height or width will be scaled to or larger than shape.
    Args:
        image: numpy image, 2d or 3d
        shape: (height, width)
    Returns:
        numpy image
    """
    height, width = shape
    imgh, imgw = image.shape[0:2]
    if imgh < height or imgw < width:
        scale = np.maximum(height/imgh, width/imgw)
        image = cv2.resize(image,(math.ceil(imgw*scale), math.ceil(imgh*scale)))  #如果图片尺寸小于裁剪尺寸，则先resize.
    return image


def np_random_crop(image, shape, random_h=None, random_w=None):
    """
    Random crop.
    Shape from image.
    Args:
        image: Numpy image, 2d or 3d.
        shape: (height, width).
        random_h: A random int.
        random_w: A random int.
    Returns:
        numpy image
        int: random_h
        int: random_w
    """
    height, width = shape
    image = np_scale_to_shape(image, shape)
    imgh, imgw = image.shape[0:2]
    if random_h is None:
        random_h = np.random.randint(imgh-height+1)
    if random_w is None:
        random_w = np.random.randint(imgw-width+1)
    return (image[random_h:random_h+height, random_w:random_w+width, :],random_h, random_w)


class InpaintDataset():
    def __init__(self, args):
        self.args = args
        self.imglist = self.get_files(args.image_dir)

    def get_files(self, path):
        ret = []
        for tuple_path in os.walk(path):
            for filespath in tuple_path[2]:
                ret.append(os.path.join(tuple_path[0], filespath))
        return ret

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        img = cv2.imread(self.imglist[index])       
        h,w = self.args.IMG_SHAPE[0],self.args.IMG_SHAPE[1]
        img = cv2.resize(img, (h,w)) 
        #img, random_h, random_w = np_random_crop(img,(h,w))      #裁剪图像的一部分区域，不是采样
        img = img / 127.5 - 1
        img = img.transpose((2, 0, 1))
        return img






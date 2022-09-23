import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import load_checkpoint, load_param_into_net
import argparse
from mindspore import Tensor
import cv2
import numpy as np
import scipy.misc
import glob 
import os
import random
import time
from scipy import signal
import progressbar
from time import sleep

from inpainting_network_mindspore import GatedGenerator
from compute_attention import *

from train1_mindspore import *

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', default='./data/test/images', type=str,
                    help='The directory of images to be completed.')
parser.add_argument('--mask_dir', default='./data/test/masks', type=str,
                    help='The directory of masks, value 255 indicates mask.')
parser.add_argument('--output_dir', default='./output', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='./model_logs/places2', type=str,
                    help='The directory of tensorflow checkpoint.')
parser.add_argument('--rectangle_mask', default=True, type=bool,
                    help='whether to use rectangle masks.')
parser.add_argument('--input_size', default=512, type=int,
                    help='The size of input image.')
parser.add_argument('--times', default=8, type=int,
                    help='The size of input image.')
parser.add_argument('--ATTENTION_TYPE', default='SOFT', type=str,
                    help='compute attention type.')
parser.add_argument('--train_batchsize', type = int, default = 1, help = ' .')
# Network parameters
parser.add_argument('--latent_channels', type = int, default = 32, help = 'latent channels')
parser.add_argument('--activation', type = str, default = 'elu', help = 'the activation type')
parser.add_argument('--norm1', type = str, default = 'none', help = 'normalization type')
parser.add_argument('--norm', type=str, default='none', help='normalization type')
parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'the initialization type')
parser.add_argument('--init_gain', type = float, default = 0.2, help = 'the initialization gain')
# Dataset parameters
#parser.add_argument('--baseroot', type = str, default = './places365_standard/train', help = 'the training folder: val_256, test_large, data_256')
parser.add_argument('--mask_type', type = str, default = 'bbox', help = 'mask type')
parser.add_argument('--mask_num', type=int, default=10, help='bbox num')
parser.add_argument('--imgsize', type = int, default = 512, help = 'size of image')
parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')
parser.add_argument('--bbox_shape', type = int, default = 30, help = 'margin of image for bbox mask')
parser.add_argument('--COARSE_ALPHA', default=1.2, type=float,
                        help='loss rate.')
parser.add_argument('--GAN_WITH_MASK', default=False, type=bool,
                    help=' .')
parser.add_argument('--GAN_LOSS_ALPHA', default=0.001, type=float,
                    help=' .')
parser.add_argument('--L1_LOSS_ALPHA', default=1.2, type=float,
                    help=' .')
parser.add_argument('--AE_LOSS_ALPHA', default=1.2, type=float,
                    help=' .')
parser.add_argument('--WGAN_GP_LAMBDA', default=10, type=int,
                        help=' .')
args = parser.parse_args()


def sort(str_lst):
    return [s for s in sorted(str_lst)]

def read_imgs_masks(args):
    paths_img = glob.glob(args.image_dir+'/*.*[g|G]')
    paths_img = sort(paths_img)
    paths_mask = glob.glob(args.mask_dir+'/*.*[g|G]')
    paths_mask = sort(paths_mask)
    return paths_img, paths_mask
    
def get_input(path_img, path_mask):
    image = cv2.imread(path_img)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(path_mask)
   
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    return np.concatenate([image, mask], axis=2), image[0], mask[0]

dtype = mindspore.float32


def build_inference_net(raw_img_ph,raw_mask_ph,model_gen,model_dis,args):
    expand_dims = ops.ExpandDims()
    raw_img = expand_dims(raw_img_ph, 0)#<class 'mindspore.common.tensor.Tensor'> (1, 1134, 2016, 3)
    raw_img = raw_img.astype(mindspore.float32)
    raw_img = ops.Transpose()(raw_img,(0,3,1,2))#NCHW<class 'mindspore.common.tensor.Tensor'> (1, 3, 1134, 2016)
    resize = ops.ResizeNearestNeighbor((args.times * args.input_size, args.times * args.input_size))
    large_img = resize(raw_img)        #(1, 3, 4096, 4096) <class 'mindspore.common.tensor.Tensor'> Float32
    large_img = ops.Reshape()(large_img,(1,3, args.times * args.input_size, args.times * args.input_size))
    large_img = large_img/127.5 - 1  #nchw  归一化到-1，1
    net = nn.Unfold([1,args.times,args.times,1],[1,args.times,args.times,1],[1,1,1,1],'same')
    small_img = net(large_img)   #提取为小块
    small_img = ops.Transpose()(small_img,(0,2,3,1)) #NHWC
    small_img = ops.Reshape()(small_img, (1, args.input_size, args.input_size, args.times, args.times, 3))  
    small_img = ops.ReduceMean(False)(small_img,axis=(3,4))  
    small_img = ops.Transpose()(small_img,(0,3,1,2)) #输出shape为NCHW的512*512大小的诸多归一化小块图

    raw_mask = expand_dims(raw_mask_ph, 0)
    raw_mask = raw_mask.astype(mindspore.float32)
    raw_mask = ops.Transpose()(raw_mask,(0,3,1,2))  #NCHW
    resize = ops.ResizeNearestNeighbor((args.input_size, args.input_size))
    small_mask = resize(raw_mask)
    small_mask = ops.Reshape()(small_mask,(1,3,args.input_size, args.input_size))
    small_mask = 1 - small_mask/255  #孔洞区域像素变为0，mask输出为NCHW 512*512的归一化图

    x2 , x2r ,corres = build_inference_graph(real=small_img,mask=small_mask,model_gen= model_gen, model_dis = model_dis,args=args)

    small_output = (x2 + 1.) * 127.5
    #small_output = ops.Transpose()(small_output,(0,2,3,1))
    #small_output = small_output.astype(mindspore.uint8) 
    #small_output = small_output[0].asnumpy()

    large_output, out1, out2, out3= post_processing(large_img, small_img, x2, small_mask, corres, args)  #残差聚合输出，原图小块上采样后，生成器输出，注意力转移后的残差
    print('post_processing')
    raw_size_output = resize_back(raw_img, large_output, small_mask) #大原图，大输出，小mask——>
    print('resize_bakc ok')
    return raw_size_output, raw_img_ph, raw_mask_ph

def build_inference_graph(real, mask, model_gen, model_dis,args):
    #input(处理后的原图，mask) NCHW 512*512
    mask = mask[0:1, 0:1, :, :]
    x = real * (1. - mask) #空洞区域为原图？？？？
    x1, x2, corres = model_gen(x, mask)
    loss_G = GenWithLossCell(model_gen,model_dis,args)(real,x,mask)
    loss_D = DisWithLossCell(model_gen,model_dis,args)(real,x,mask)
    print('output of model',x1.shape, x2.shape, corres.shape)#Tensor("generator/ResizeBilinear_1:0", shape=(1, 512, 512, 3), dtype=float32) Tensor("generator/clip_by_value_1:0", shape=(1, 512, 512, 3), dtype=float32) Tensor("generator/concat_3:0", shape=(1, 32, 32, 1024), dtype=float32)
    print(loss_G, loss_D)
    fake = x2
    fake_patched = fake * mask + x * (1-mask)
    return x2, fake_patched, corres   #NCHW

    
def gaussian_kernel(size, std):
    k = signal.gaussian(size, std)  #大小size*size,方差std，均值默认0
    kk = np.matmul(k[:, np.newaxis], [k])
    return kk/np.sum(kk)

def resize_back(raw_img, large_output, small_mask):
    raw_shp = raw_img.shape  #NCHW
    resize = nn.ResizeBilinear()
    raw_size_output = resize(large_output,size=(raw_shp[2], raw_shp[3])) #将图像采样到原始输入大小
    raw_size_output = raw_size_output.astype(dtype)

    gauss_kernel = gaussian_kernel(7,  1.)
    gauss_kernel = Tensor(gauss_kernel)
    gauss_kernel = gauss_kernel.astype(dtype)
    gauss_kernel = ops.ExpandDims()(gauss_kernel,2)
    gauss_kernel = ops.ExpandDims()(gauss_kernel,3)    
    
    a,b,c,d = ops.Shape()(gauss_kernel)
    gauss_kernel = ops.Transpose()(gauss_kernel,(3,2,0,1))
    conv = nn.Conv2d(c,d,(a,b),1,pad_mode='same',padding = 0,weight_init=gauss_kernel,data_format='NCHW')#对mask进行高斯卷积
    mask = conv(small_mask[:,0:1,:,:])
    resize = nn.ResizeBilinear()
    mask = resize(mask,size=(raw_shp[2], raw_shp[3]))
    mask = mask.astype(dtype)  #f32
    raw_size_output = raw_size_output * mask + raw_img * (1-mask) #f32
    raw_size_output = ops.Transpose()(raw_size_output,(0,2,3,1))  #NHWC   F32
    raw_size_output = raw_size_output.astype(mindspore.uint8)
    return raw_size_output

def post_processing(large_img, small_img, low_base, small_mask, corres, args):
    high_raw = large_img
    low_raw = small_img
    mask = 1 - small_mask
    resize = nn.ResizeBilinear()
    low_raw = resize(low_raw,scale_factor=args.times)
    to_shape = list(ops.Shape()(mask))[2:]  
    to_shape[0], to_shape[1] = int(to_shape[0] * args.times), int(to_shape[1] * args.times)
    resize = ops.ResizeNearestNeighbor((to_shape[0], to_shape[1]))  
    mask = resize(mask)
    residual1 = (high_raw - low_raw) * mask  #高频残差
    print('residual1',residual1.shape,corres.shape)
    residual = apply_attention2([1,3,4096,4096],[1,1024,32,32])(residual1, corres)
    resize = nn.ResizeBilinear()
    low_base = resize(low_base,scale_factor=args.times)
    x = low_base + residual
    x = x.clip(-1,1)
    x = (x + 1.) * 127.5
    return x, low_raw, low_base, residual 


from mindspore import context
if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE)
    paths_img, paths_mask = read_imgs_masks(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    total_time = 0.

    bar = progressbar.ProgressBar(maxval=len(paths_img), \
         widgets=[progressbar.Bar('=', '[', ']'), ' ', \
         progressbar.Percentage()])
    bar.start()
    model_dis = Discriminator(args)
    model_gen = GatedGenerator(args)
    #param_dict = load_checkpoint('./cra1.ckpt')
    param_dict = load_checkpoint("/data0/cra_ckpt/place_resize_lr7/generator_epoch6_batch56358.ckpt")
    load_param_into_net(model_gen, param_dict)
    load_param_into_net(model_dis, param_dict)
    for (i, path_img) in enumerate(paths_img):
        rint = i % len(paths_mask)
        bar.update(i+1)
        in_img, img, mask = get_input(path_img, paths_mask[rint])
        s = time.time()

        raw_img_ph = Tensor(img)
        raw_mask_ph = Tensor(255 - mask)
        #print(raw_img_ph.dtype,raw_mask_ph.dtype)  # uint8,uint8
        outputs, raw_img_ph, raw_mask_ph = build_inference_net(raw_img_ph,raw_mask_ph,model_gen,model_dis,args)#(1, 3, 1134, 2016) (1134, 2016, 3) (1134, 2016, 3)
        print(type(outputs),type(raw_img_ph),type(raw_mask_ph))
        res = outputs[0]
        res = res.asnumpy()
        #res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        print('res',type(res))
        total_time += time.time() - s
        img_hole = img * (1-mask/255) + mask 
        #print(img_hole.shape)
        print('type',type(res),type(img),type(mask))
        res = np.concatenate([img, img_hole, res], axis=1)  #数组拼接  将三张图片显示在一张图片中
        print('concat ok')
        cv2.imwrite(args.output_dir + '/' + str(i)+ '.jpg', res)
        print('save')

    bar.finish()
    print('average time per image', total_time/len(paths_img))

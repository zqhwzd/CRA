import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from network_module_mindspore import *
import numpy as np
from compute_attention import *
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint,load_checkpoint


class Coarse(nn.Cell):
    def __init__(self,opt):
        super(Coarse,self).__init__()
        self.coarse1 = nn.SequentialCell(
            GatedConv2d(4, 32, 5, 2, 1, activation=opt.activation, sc=True),#name = 'c_en_down_128_feat/gate'
            GatedConv2d(32, 32, 3, 1, 1, activation=opt.activation, sc=True),#name = 'c_en_conv_128_feat/gate'
            GatedConv2d(32, 64, 3, 2, 1, activation=opt.activation, sc=True)#name = 'c_en_down_64_feat/gate'
        )
        self.coarse2 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, sc=True),#name = 'c_en_conv1_64_feat/gate'
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, sc=True),#name = 'c_en_conv2_64_feat/gate'
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, sc=True)#name = 'c_en_conv3_64_feat/gate'
        )
        self.coarse3 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, sc=True),#name ='c_dil_d1_feat'
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, sc=True),#name ='c_dil_d2_feat'
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, sc=True)#name ='c_dil_d3_feat'
        )
        self.coarse4 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 2, activation=opt.activation, sc=True),#name ='c_dil_d4_feat'
            GatedConv2d(64, 64, 3, 1, 2, activation=opt.activation, sc=True),#name ='c_dil_d5_feat'
            GatedConv2d(64, 64, 3, 1, 2,activation=opt.activation, sc=True),#name ='c_dil_d6_feat'
            GatedConv2d(64, 64, 3, 1, 2,activation=opt.activation, sc=True),#name ='c_dil_d7_feat'
            GatedConv2d(64, 64, 3, 1, 2,  activation=opt.activation, sc=True)#name ='c_dil_d8_feat'
        )
        self.coarse5 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 4, activation=opt.activation, sc=True),#name ='c_dil_d9_feat'
            GatedConv2d(64, 64, 3, 1, 4, activation=opt.activation, sc=True),#name ='c_dil_d10_feat'
            GatedConv2d(64, 64, 3, 1, 4, activation=opt.activation, sc=True),#name ='c_dil_d11_feat'
            GatedConv2d(64, 64, 3, 1, 4, activation=opt.activation, sc=True)#name ='c_dil_d12_feat'
        )
        self.coarse6 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 8,activation=opt.activation, sc=True),#name ='c_dil_d13_feat'
            GatedConv2d(64, 64, 3, 1, 8,activation=opt.activation, sc=True),#name ='c_dil_d14_feat'
        )
        self.coarse7 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, sc=True),#name='c_de_conv1_64_feat/gate'
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, sc=True),#name='c_de_conv2_64_feat/gate'
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation, sc=True),#name='c_de_conv3_64_feat/gate'
        )
        self.coarse8 = nn.SequentialCell(
            TransposeGatedConv2d(64, 32, 3, 1, 1, activation=opt.activation, sc=True),#name='c_de_up_128_conv_feat/gate'
            GatedConv2d(32, 32, 3, 1, 1, activation=opt.activation,sc=True),#name='c_de_conv_128_feat/gate'
            TransposeGatedConv2d(32, 3, 3, 1, 1, activation=opt.activation, sc=True),# name='c_de_toRGB_conv_feat/gate'
        )

    def construct(self, first_in):
        first_out = self.coarse1(first_in)
        first_out = self.coarse2(first_out)
        first_out = self.coarse3(first_out)
        first_out = self.coarse4(first_out)
        first_out = self.coarse5(first_out) 
        first_out = self.coarse6(first_out)
        first_out = self.coarse7(first_out) 
        first_out = self.coarse8(first_out)
        first_out = ops.clip_by_value(first_out, -1, 1)
        return first_out


class GatedGenerator(nn.Cell):
    def __init__(self, opt):
        super(GatedGenerator, self).__init__()
        self.coarse = Coarse(opt)
        self.refinement1 = nn.SequentialCell(
            GatedConv2d(4, 32, 3, 2, 1, activation=opt.activation), #name='re_en_down_256'              
            GatedConv2d(32, 32, 3, 1, 1, activation=opt.activation)#name='re_en_conv_256' 
        )
        self.refinement2 = nn.SequentialCell(
            GatedConv2d(32, 64, 3, 2, 1, activation=opt.activation),#name='re_en_down_128'  
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation)#name='re_en_conv_128' 
        )
        self.refinement3 = nn.SequentialCell(
            GatedConv2d(64, 128, 3, 2, 1, activation=opt.activation),#name='re_en_down_64' 
            GatedConv2d(128, 128, 3, 1, 1, activation=opt.activation)#name='re_en_conv_64'  
        )
        self.refinement4 = GatedConv2d(128, 128, 3, 1, 1, activation=opt.activation)#name='re_dil_d1'
        self.refinement5 = nn.SequentialCell(
            GatedConv2d(128, 128, 3, 1, 2,activation=opt.activation),#name='re_dil_d2'
            GatedConv2d(128, 128, 3, 1, 4,activation=opt.activation)#name='re_dil_d4'
        )
        self.refinement6 = nn.SequentialCell(
            GatedConv2d(128, 128, 3, 1, 8, activation=opt.activation),#name='re_dil_d8'
            GatedConv2d(128, 128, 3, 1, 16, activation=opt.activation)#name='re_dil_d16'
        )
        self.refinement7 = nn.SequentialCell(
            TransposeGatedConv2d(128, 64, 3, 1, 1, activation=opt.activation),#name='re_de_up__128_conv'
            GatedConv2d(64, 64, 3, 1, 1, activation=opt.activation)#name='re_de_conv_128'
        )
        self.refinement8 = nn.SequentialCell( 
            TransposeGatedConv2d(128, 32, 3, 1, 1, activation=opt.activation),#name='re_de_up__256_conv'
            GatedConv2d(32, 32, 3, 1, 1, activation=opt.activation)#name='re_de_conv_256'
        )
        self.refinement9 = TransposeGatedConv2d(64, 3, 3, 1, 1, activation=opt.activation)#name='re_de_toRGB__256_conv'
        self.conv_att1 = GatedConv2d(128, 128, 3, 1, 1, activation=opt.activation)#name='re_att_64_att1'
        self.conv_att2 = GatedConv2d(256, 128, 3, 1, 1, activation=opt.activation)#name='re_att_64_att3' 
        self.batch = opt.train_batchsize
        self.apply_attention1 = apply_attention([self.batch,64,128,128],[self.batch,1024 ,32,32])
        self.apply_attention2 = apply_attention([self.batch,32,256,256],[self.batch,1024 ,32,32])
        self.ones = ops.Ones()
        self.concat = ops.Concat(1)
        self.bilinear_256 = ops.ResizeBilinear((256,256))
        self.bilinear_512 = ops.ResizeBilinear((512,512))
        self.reshape = ops.Reshape()
        self.contextual_attention = contextual_attention(fuse=True,dtype=mindspore.float32)
        self.cat = ops.Concat(1)
    def construct(self, img, mask):
        x_in = img.astype(mindspore.float32)
        shape = x_in.shape #NCHW
        #print(shape)
        mask_batch = self.ones((shape[0],1,shape[2],shape[3]), mindspore.float32) 
        mask_batch = mask_batch * mask
        first_in = self.concat((x_in, mask_batch)) 
        first_in = self.bilinear_256(first_in)
        first_out = self.coarse(first_in)  
        first_out = self.bilinear_512(first_out)  #(1, 3, 512, 512) <class 'mindspore.common.tensor.Tensor'>    
        first_out = self.reshape(first_out,(shape[0],shape[1],shape[2],shape[3]))
        #print(mask_batch.shape,type(mask_batch))
        x_coarse = first_out * mask_batch + x_in * (1.- mask_batch)
        #print('coarse ok')
        second_in = self.concat([x_coarse, mask_batch])  #(1, 4, 512, 512)
        pl1 = self.refinement1(second_in)                   
        pl2 = self.refinement2(pl1)                        
        second_out = self.refinement3(pl2) 
        second_out = self.refinement4(second_out)               
        second_out = self.refinement5(second_out)
        pl3 = self.refinement6(second_out)  
        #second_out,match=self.apply_contextual_attention(pl3,mask)
        x_hallu = pl3
        x, match = self.contextual_attention(pl3, pl3, mask)
        x = self.conv_att1(x)
        x = self.cat((x_hallu, x))
        second_out = self.conv_att2(x)
        #print(match.shape,second_out.shape)  (1,1024 ,32 , 32) (1, 128, 64, 64)
        shp_att = match.shape
        second_out = self.refinement7(second_out)    
        shp_pl2 = pl2.shape 
        second_out_att = self.apply_attention1(pl2,match)
        second_out = self.concat([second_out_att,second_out]) #1,128,128,128
        second_out = self.refinement8(second_out) 
        shp_pl1 = pl1.shape
        second_out_att = self.apply_attention2(pl1,match)
        second_out = self.concat([second_out_att,second_out]) 
        second_out = self.refinement9(second_out) 
        second_out = ops.clip_by_value(second_out, -1, 1)
        return first_out, second_out,match


class Discriminator(nn.Cell):
    def __init__(self, args):
        super(Discriminator, self).__init__()     
        self.block1 = Conv2dLayer(3, 64, 5, 2, 1) #name='conv1'
        self.block2 = Conv2dLayer(64, 128, 5, 2, 1)#name='conv2'
        self.block3 = Conv2dLayer(128, 256, 5, 2, 1)#name='conv3'
        self.block4 = Conv2dLayer(256, 256, 5, 2, 1)#name='conv4'
        self.block5 = Conv2dLayer(256, 256, 5, 2, 1)#name='conv5'
        self.block6 = Conv2dLayer(256, 256, 5, 2, 1)#name='conv6'
        self.block7 = nn.Dense(16384,1)#name='linear'
    def construct(self, img):
        x = img
        x = self.block1(x)  
        x = self.block2(x)  
        x = self.block3(x)  
        x = self.block4(x) 
        x = self.block5(x) 
        x = self.block6(x)  
        x = x.reshape([x.shape[0], -1])  
        x = self.block7(x) 
        return x


import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batchsize', type = int, default = 4, help = 'size of the batches')
    parser.add_argument('--activation', type = str, default = 'elu', help = 'the activation type')
    parser.add_argument('--norm1', type = str, default = 'none', help = 'normalization type')
    parser.add_argument('--norm', type=str, default='none', help='normalization type')
    parser.add_argument('--ATTENTION_TYPE', default='SOFT', type=str,
                    help='compute attention type.')
    return parser.parse_args()

if __name__ == "__main__":
    opt = get_args()
    gen = GatedGenerator(opt)
    dis = Discriminator(opt)
    #pf = open('E:/try3.txt', 'w+') 
    #for i in gen.parameters_and_names():
     #   print(i)
        #pf.write(str(i))
        #pf.write("\n")
    #for i in dis.parameters_and_names():
      #  print(i)
        #pf.write(str(i))
        #pf.write("\n")
    a = load_checkpoint('./cra1.ckpt')
    for key,value in a.items():
        print(key)
        print(value)



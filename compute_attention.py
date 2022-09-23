import mindspore
import mindspore.nn as nn 
import mindspore.ops as ops
import numpy as np
from mindspore.ops import constexpr
from network_module_mindspore import GatedConv2d

def downsample(x):
    shp = x.shape
    net = nn.Unfold([1,1,1,1], [1,2,2,1], [1,1,1,1],'same')
    x = net(x)   #NCHW
    return ops.Reshape()(x, (shp[0],shp[1],shp[2]//2,shp[3]//2))


class contextual_attention(nn.Cell):
    def __init__(self,fuse_k=3,softmax_scale=10,fuse=True,dtype=mindspore.float32):
        super(contextual_attention,self).__init__()
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.dtype = dtype
        self.reducesum = ops.ReduceSum(False)
        self.unfold1 = nn.Unfold([1,3,3,1],[1,2,2,1],[1,1,1,1],'same')
        self.unfold2 = nn.Unfold([1,3,3,1],[1,1,1,1],[1,1,1,1],'same')
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.pool1 = nn.MaxPool2d(16,16,'same','NCHW')
        self.pool2 = nn.MaxPool2d(3,1,'same','NCHW')
        self.maximum = ops.Maximum()
        self.sqrt = ops.Sqrt()
        self.square = ops.Square()
        self.eye = ops.Eye()
        self.reducemax = ops.ReduceMax(True)
        self.greaterequal = ops.GreaterEqual()
        self.pow = ops.Pow()
        self.div = ops.Div()
        self.softmax = nn.Softmax(1)
        self.cat = ops.Concat(0)
        self.conv1 = InitConv2d([3,3,128,1024],1,True)
        self.conv2 = InitConv2d([3,3,1,1],1,True)
        self.disconv1 = InitConv2d([3,3,128,1024],2,False)

    def construct(self,src, ref,mask,method='SOFT'):
        shape_src = src.shape
        shape_ref = ref.shape
        batch_size = shape_src[0]   #1
        nc = shape_src[1]            #128 
        raw_feats = self.unfold1(ref)#NCHW
        raw_feats = self.transpose(raw_feats,(0,2,3,1)) #NHWC
        raw_feats = self.reshape(raw_feats, (batch_size, -1, 3,3, nc))  #1,1922,3,3,64
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 4, 1))
        split = ops.Split(0,batch_size)
        raw_feats_lst = split(raw_feats) #1,3,3,64,1922 nhwc
        src = downsample(src) #NCHW
        ref = downsample(ref)  #NCHW
        ss = src.shape#tuple
        rs = ref.shape
        src_lst = split(src)  #nchw
        feats = self.unfold2(ref)
        feats = self.transpose(feats,(0,2,3,1))  #NHWC
        feats = self.reshape(feats, (batch_size, -1, 3,3, nc))
        feats = self.transpose(feats,(0,2,3,4,1))
        feats_lst = split(feats)  #nhwc
        mask = self.pool1(mask)
        mask = self.pool2(mask)
        mask = 1-mask
        mask = self.reshape(mask, (1, -1, 1, 1)) #NCHW        
        y_lst, y_up_lst = [], []
        offsets = []
        fuse_weight = self.reshape(self.eye(3,3,mindspore.float32),(3,3,1,1))
        for x, r, raw_r in zip(src_lst, feats_lst, raw_feats_lst):
            r = r[0]  #nhwc
            r = r / self.maximum(self.sqrt(self.reducesum(self.square(r),[0,1,2])),1e-8) #r:3，3，128，1024   x:1,32,32,128
            r_kernel = self.transpose(r,(3,2,0,1))
            y = self.conv1(x,r_kernel)
            if self.fuse:
                yi = self.reshape(y, (1, 1,ss[2]*ss[3], rs[2]*rs[3])) #NCHW 1,1,1024,1024
                fuse_weight_kernel = ops.Transpose()(fuse_weight,(3,2,0,1))
                yi = self.conv2(yi,fuse_weight_kernel)
                yi = self.transpose(yi,(0,2,3,1)) #NHWC
                yi = self.reshape(yi,(1, ss[2], ss[3], rs[2], rs[3]))
                yi = self.transpose(yi, (0, 2, 1, 4, 3))
                yi = self.reshape(yi, (1, ss[2]*ss[3], rs[2]*rs[3], 1))
                yi = self.transpose(yi,(0,3,1,2))
                yi = self.conv2(yi,fuse_weight_kernel)
                yi = self.transpose(yi,(0,2,3,1)) #NHWC
                yi = self.reshape(yi, (1, ss[3], ss[2], rs[3], rs[2]))
                yi = self.transpose(yi,(0,2,1,4,3))
                y = yi       
            y = self.reshape(y, (1, ss[2], ss[3], rs[2]*rs[3]))#nhwc
            y = self.transpose(y,(0,3,1,2))  #NCHW
            if method == 'HARD':
                ym = self.reducemax(y,1)
                y = y * mask
                coef = self.greaterequal(y,max(y,1)).astype(self.dtype)
                y = self.pow(coef * self.div(y, ym + 1e-04 ), 2)
            elif method == 'SOFT':
                y = (self.softmax(y * mask * self.softmax_scale))*mask
            y = self.reshape(y,(1, rs[2]*rs[3],ss[2], ss[3]))  #NCHW
            if self.dtype == mindspore.float32:
                offset = y.argmax(1)  #C
                offsets.append(offset)              
            feats = raw_r[0]  #NHWC
            #print('y',y.shape,feats.shape)
            feats_kernel = self.transpose(feats,(3,2,0,1))
            y_up = self.disconv1(y,feats_kernel)
            y_lst.append(y)  #NCHW
            y_up_lst.append(y_up)  #NCHW
    
        out,correspondence = self.cat(y_up_lst),self.cat(y_lst)  #NCHW
        #print('out',out.shape,correspondence.shape,shape_src)
        out = self.reshape(out,(shape_src[0],shape_src[1],shape_src[2],shape_src[3]))
        #print('out',out.shape)
        return out, correspondence

def freeze(layer):
    for param in layer.get_parameters():
        param.requires_grad = False

class InitConv2d(nn.Cell):
    def __init__(self,shape,rate=1,con_dis=True):
        super(InitConv2d,self).__init__()
        self.shape = shape
        self.rate = rate
        self.con_dis = con_dis
        self.H,self.W,self.I,self.O = self.shape[0],self.shape[1],self.shape[2],self.shape[3]
        if self.con_dis:
            self.conv = nn.Conv2d(self.I,self.O,(self.H,self.W),(1,1),'same')
            freeze(self.conv)
        else:
            self.conv = nn.Conv2dTranspose(self.O,self.I,(self.H,self.W),(self.rate,self.rate),'same')
            freeze(self.conv)
        self.tmp = mindspore.ParameterTuple(self.get_parameters()) 
        
    def construct(self, x, w):
        for weight in self.tmp:
            ops.Assign()(weight, w)
        return self.conv(x)

"""
class apply_contextual_attention(nn.Cell):
    def __init__(self, conv_func1,conv_func2,method = 'SOFT', dtype=mindspore.float32):
        super(apply_contextual_attention,self).__init__()
        self.method = method
        self.cat = ops.Concat(1)
        self.dtype = dtype
        self.conv_func1 = conv_func1
        self.conv_func2 = conv_func2
        self.contextual_attention = contextual_attention(fuse=True,dtype=self.dtype)
    def construct(self,x, mask_s):
        method = self.method
        x_hallu = x
        x, corres = self.contextual_attention(x, x, mask_s, method = method)
        x = self.conv_func1(x)
        x = self.cat((x_hallu, x))
        x = self.conv_func2(x)
        return x, corres
"""


class apply_attention(nn.Cell):
    def __init__(self,shp,shp_att):
        super(apply_attention,self).__init__()
        self.shp = shp
        self.shp_att = shp_att
        self.rate = self.shp[2] // self.shp_att[2]
        self.kernel = self.rate * 2
        self.batch_size = self.shp[0]
        self.sz = self.shp[2]   #h
        self.nc = self.shp[1]  #c
        self.unfold = nn.Unfold([1,self.kernel,self.kernel,1], [1,self.rate,self.rate,1], [1,1,1,1],'same')
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.split = ops.Split(0,self.batch_size)
        self.disconv1 = InitConv2d([8,8,64,1024],self.rate,False)
        self.disconv2 = InitConv2d([16,16,32,1024],self.rate,False)
        self.concat = ops.Concat(0)
        self.conv_pl2 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 1),#name='re_de_att_128_1'
            GatedConv2d(64, 64, 3, 1, 2)#name='re_de_att_128_2'
        )
        self.conv_pl1 = nn.SequentialCell(
            GatedConv2d(32, 32, 3, 1, 1),#name='re_de_att_256_1'
            GatedConv2d(32, 32, 3, 1, 2)#name='re_de_att_256_2'
        )
    def construct(self,x, correspondence):
        raw_feats = self.unfold(x)  #NCHW    pl2-b,4096,32,32     pl1-b,8192,32,32
        raw_feats = self.transpose(raw_feats,(0,2,3,1))  #NHWC
        raw_feats = self.reshape(raw_feats, (self.batch_size, -1, self.kernel, self.kernel, self.nc))  #1,1024,8,8,64/1,1024,16,16,32
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 4, 1))  #1,8,8,64,961/1,16,16,32,961
        raw_feats_lst = self.split(raw_feats)   #NHWC     
        ys = []
        correspondence = self.transpose(correspondence,(0,2,3,1)) #NHWC
        att_lst = self.split(correspondence)  #NHWC
        for feats, att in zip(raw_feats_lst, att_lst):   
            feats_kernel = self.transpose(feats[0],(3,2,0,1)) 
            att = self.transpose(att,(0,3,1,2))
            if self.shp[2] == 128:
                y1 = self.disconv1(att,feats_kernel)
                ys.append(y1)
            elif self.shp[2] == 256:
                y2 = self.disconv2(att,feats_kernel)
                ys.append(y2)
            else:
                print('Value Error')
        out = self.concat(ys)#NCHW
        if self.shp[2] == 128:
            out = self.conv_pl2(out)
        elif self.shp[2] == 256:
            out = self.conv_pl1(out)
        else:
            print('conv error')
        return out


class apply_attention2(nn.Cell):
    def __init__(self,shp,shp_att):
        super(apply_attention2,self).__init__()
        self.shp = shp
        self.shp_att = shp_att
        self.rate = self.shp[2] // self.shp_att[2]
        self.kernel = self.rate
        self.batch_size = self.shp[0]
        self.sz = self.shp[2]   #h
        self.nc = self.shp[1]  #c
        self.unfold = nn.Unfold([1,self.kernel,self.kernel,1], [1,self.rate,self.rate,1], [1,1,1,1],'same')
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.split = ops.Split(0,self.batch_size)
        self.disconv1 = InitConv2d([128,128,3,1024],self.rate,False)
        self.concat = ops.Concat(0)

    def construct(self,x, correspondence):
        print('index',self.shp_att,self.shp)
        raw_feats = self.unfold(x)  #NCHW    pl2-b,4096,32,32     pl1-b,8192,32,32
        raw_feats = self.transpose(raw_feats,(0,2,3,1))  #NHWC
        raw_feats = self.reshape(raw_feats, (self.batch_size, -1, self.kernel, self.kernel, self.nc))  #1,1024,8,8,64/1,1024,16,16,32
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 4, 1))  #1,8,8,64,961/1,16,16,32,961
        raw_feats_lst = self.split(raw_feats)   #NHWC     
        ys = []
        correspondence = self.transpose(correspondence,(0,2,3,1)) #NHWC
        att_lst = self.split(correspondence)  #NHWC
        for feats, att in zip(raw_feats_lst, att_lst):   
            feats_kernel = self.transpose(feats[0],(3,2,0,1)) 
            print('feats',feats_kernel.shape)
            att = self.transpose(att,(0,3,1,2))
            y = self.disconv1(att,feats_kernel)
            ys.append(y)
        out = self.concat(ys)#NCHW
        return out



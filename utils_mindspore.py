import numpy as np
import mindspore.ops as ops
import mindspore
from mindspore.ops import constexpr
from mindspore import nn
from mindspore.ops.composite import GradOperation


def gan_wgan_loss(pos,neg):
    d_loss = ops.ReduceMean(False)(neg) - ops.ReduceMean(False)(pos)
    g_loss = -ops.ReduceMean(False)(neg)
    return g_loss,d_loss

@constexpr
def generate_tensor0():
    return mindspore.Tensor(0, mindspore.float32)

@constexpr
def generate_tensor1():
    return mindspore.Tensor(1, mindspore.float32)

def random_interpolates(pos,neg):
    minval = generate_tensor0()
    maxval = generate_tensor1()
    epsilon = ops.uniform((pos.shape[0],1,1,1),minval,maxval,dtype=mindspore.float32)
    X_hat = pos + epsilon * (neg - pos)
    return X_hat


"""
class Net(nn.Cell):
    def __init__(self,net):
        super(Net,self).__init__()
        self.net = net
    def construct(self,x):
        out = self.net(x)
        return out
class GradNet(nn.Cell):
    def __init__(self,net):
        super(GradNet,self).__init__()
        self.net = net
        self.net = Net(self.net)
        self.grad_op = GradOperation(get_all=False)
    def construct(self,x):      
        gradient_function = self.grad_op(self.net)
        return gradient_function(x)
"""


class gradients_penalty(nn.Cell):
    def __init__(self,netD):
        super(gradients_penalty,self).__init__()
        #self.gradients = GradNet(netD)
        self.sqrt = ops.Sqrt()
        self.reducesum = ops.ReduceSum()
        self.square = ops.Square()
        self.reducemean = ops.ReduceMean()
        self.gradients = GradOperation(get_all=False)(netD)
    def construct(self,interpolates_global):
        grad_D_X_hat = self.gradients(interpolates_global)
        slopes = self.sqrt(self.reducesum(self.square(grad_D_X_hat),[1,2,3]))
        gradients_penalty = self.reducemean((slopes - 1) ** 2)
        return gradients_penalty




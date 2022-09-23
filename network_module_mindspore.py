import mindspore
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal


class Conv2dLayer(nn.Cell):
    """
        Define the convolutional layer in the discriminator
        including convolution, activation, normalization, and padding operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(Conv2dLayer, self).__init__()
        self.activation = nn.LeakyReLU(0.2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same', dilation=dilation, has_bias=True,weight_init=TruncatedNormal(0.05))
    
    def construct(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        return x


class depth_separable_conv(nn.Cell):
    """
        Building gate branch of depth-separable LWGC.
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation):
        super(depth_separable_conv, self).__init__()
        self.ds_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride,
                                    pad_mode='same', padding=0,dilation=dilation, group=1, has_bias=True,weight_init=TruncatedNormal(0.05))

    def construct(self, x):
        x = self.ds_conv(x)
        return x


class sc_conv(nn.Cell):
    """
        Building gate branch of single-channel LWGC.
    """
    def __init__(self, in_channel, out_channel, kernel_size, stride, pad_mode,padding, dilation):
        super(sc_conv, self).__init__()
        self.single_channel_conv = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=kernel_size,
                                             stride=stride, pad_mode='same', padding=padding, dilation=dilation, group=1,
                                             has_bias=True,weight_init=TruncatedNormal(0.05))

    def construct(self, x):
        x = self.single_channel_conv(x)
        return x


class GatedConv2d(nn.Cell):
    """
        Implement complete depth-separable and single-channel LWGC operation.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride, dilation, activation='elu', sc=False):
        super(GatedConv2d, self).__init__()
        if activation == 'elu':
            self.activation = nn.ELU(alpha=1.0)
        if sc:
            self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode='same', padding=0,
                                    dilation=dilation, has_bias=True,weight_init=TruncatedNormal(0.05))
            self.gate_factor = sc_conv(in_channel, 1, kernel_size, stride,pad_mode='same', padding=0, dilation=dilation)
        else:
            self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode='same', padding=0,
                                    dilation=dilation, has_bias=True,weight_init=TruncatedNormal(0.05))
            self.gate_factor = depth_separable_conv(in_channel, out_channel, 1, stride, padding=0,
                                                    dilation=dilation)

        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        gc_f = self.conv2d(x)
        gc_g = self.gate_factor(x)
        x = self.sigmoid(gc_g) * self.activation(gc_f)
        return x


class TransposeGatedConv2d(nn.Cell):
    """
        Add upsampling operation to gated convolution.
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride, dilation=1, 
                 activation='elu', sc=False, scale_factor=2):
        super(TransposeGatedConv2d, self).__init__()
        self.scale_factor = scale_factor
        self.gate_conv2d = GatedConv2d(in_channel, out_channel, kernel_size, stride, dilation,
                                       activation, sc)

    def construct(self, x):
        x = nn.ResizeBilinear()(x, scale_factor=self.scale_factor)
        x = self.gate_conv2d(x)
        return x





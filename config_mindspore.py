import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',help='The directory of images to be completed.')
    parser.add_argument('--mask_template_dir',help='The directory of masks, value 255 indicates mask.')
    parser.add_argument('--IMG_SHAPE', default=[512,512,3], 
                        help='Where to write output.')
    parser.add_argument('--workers', default=1, type=int,
                        help='The num of threads')
    parser.add_argument('--input_size', default=512, type=int,
                        help='The size of input image.')
    parser.add_argument('--times', default=8, type=int,
                        help='The size of input image.')
    parser.add_argument('--ATTENTION_TYPE', default='SOFT', type=str,
                        help='compute attention type.')
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
    parser.add_argument('--learning_rate', type = float, default =1e-4, help = ' learning rate')
    parser.add_argument('--device_target', type = str, default = 'GPU', help = ' .')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 1, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--activation', type = str, default = 'elu', help = 'the activation type')
    parser.add_argument('--train_batchsize', type = int, default = 4, help = ' .')
    parser.add_argument('--Epochs', type = int, default = 20, help = ' .')
    parser.add_argument('--dis_iter', type = int, default = 1, help = ' .')
    parser.add_argument('--save_folder',help = ' .')
    return parser.parse_args(args=[])

cra_config = parse_args()






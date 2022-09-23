import os
import numpy as np
from mask_mindspore import random_mask
from config_mindspore import cra_config
from utils_mindspore import *
from dataset_mindspore import *
from inpainting_network_mindspore import *
import mindspore.dataset as ds
from mindspore import nn,Tensor
from mindspore.common import dtype as mstype
from mindspore import context,save_checkpoint
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.ops.composite as C
from mindspore.parallel._utils import (_get_device_num,_get_gradients_mean,_get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
#from mindspore.communication.management import init, get_group_size
#from device_adapter import get_device_num, get_rank_id, get_device_id
from mindspore.communication import get_rank, get_group_size
from mindspore.communication import init


class GenWithLossCell(nn.Cell):
    def __init__(self,netG,netD,args,auto_prefix=True):
        super(GenWithLossCell,self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netD = netD
        self.COARSE_ALPHA = args.COARSE_ALPHA
        self.GAN_WITH_MASK = args.GAN_WITH_MASK
        self.GAN_LOSS_ALPHA = args.GAN_LOSS_ALPHA
        self.L1_LOSS_ALPHA = args.L1_LOSS_ALPHA
        self.AE_LOSS_ALPHA = args.AE_LOSS_ALPHA
        self.train_batchsize = args.train_batchsize
        self.mean = ops.ReduceMean(False)
        self.abs = ops.Abs()
        self.concat_0 = ops.Concat(0)
        self.concat_3 = ops.Concat(3)
        self.split = ops.Split(0,2)
        self.gan_wgan_loss = gan_wgan_loss
        self.tile = ops.Tile()

    def construct(self,real,x,mask):
        x1,x2,match = self.netG(real,mask)
        fake = x2
        losses = {}
        fake_patched = fake * mask + real * (1-mask)
        fake_patched = fake_patched.astype(mindspore.float32)
        coarse_alpha = self.COARSE_ALPHA
        losses['l1_loss'] = coarse_alpha * self.mean(self.abs(real - x1) * mask)
        losses['l1_loss'] = losses['l1_loss'] + self.mean(self.abs(real - x2) * mask)
        losses['ae_loss'] = coarse_alpha * self.mean(self.abs(real - x1) * (1-mask))
        losses['ae_loss'] = losses['ae_loss'] + self.mean(self.abs(real - x2) * (1-mask))
        losses['ae_loss'] = losses['ae_loss'] / self.mean(1 - mask)
        real_fake = self.concat_0((real,fake_patched))
        if self.GAN_WITH_MASK:
            real_fake = self.concat_3((real_fake,self.tile(mask,(self.train_batchsize*2, 1, 1, 1))))
        D_real_fake = self.netD(real_fake)
        D_real,D_fake = self.split(D_real_fake)
        g_loss,d_loss = self.gan_wgan_loss(D_real,D_fake)
        losses['adv_gloss'] = g_loss
        #print('first gloss',losses['g_loss'])
        losses['g_loss'] = self.GAN_LOSS_ALPHA * losses['adv_gloss']
        losses['g_loss'] = losses['g_loss'] + self.L1_LOSS_ALPHA * losses['l1_loss']
        losses['g_loss'] = losses['g_loss'] + self.AE_LOSS_ALPHA * losses['ae_loss']
        #print('l1 loss',losses['l1_loss'], 'ae loss',losses['ae_loss'])
        loss_G = losses['g_loss']
        return loss_G
        

class DisWithLossCell(nn.Cell):
    def __init__(self,netG,netD,args,auto_prefix=True):
        super(DisWithLossCell,self).__init__(auto_prefix=auto_prefix)
        self.netG = netG
        self.netD = netD
        self.GAN_WITH_MASK = args.GAN_WITH_MASK
        self.WGAN_GP_LAMBDA = args.WGAN_GP_LAMBDA
        self.train_batchsize = args.train_batchsize
        self.concat_0 = ops.Concat(0)
        self.concat_3 = ops.Concat(3)
        self.split = ops.Split(0,2)
        self.gan_wgan_loss = gan_wgan_loss
        self.random_interpolates = random_interpolates
        self.gradients_penalty = gradients_penalty(self.netD)

    def construct(self,real,x,mask):
        x1,x2,match = self.netG(real,mask)
        fake = x2
        losses = {}
        fake_patched = fake * mask + real * (1-mask)
        fake_patched = fake_patched.astype(mindspore.float32)
        real_fake = self.concat_0((real,fake_patched))
        if self.GAN_WITH_MASK:
            real_fake = self.concat_3((real_fake,ops.Tile()(mask,(self.train_batchsize*2, 1, 1, 1))))
        D_real_fake = self.netD(real_fake)
        D_real,D_fake = self.split(D_real_fake)
        g_loss,d_loss = self.gan_wgan_loss(D_real,D_fake)
        losses['adv_dloss'] = d_loss
        #print('first dloss', losses['d_loss'])
        interps = self.random_interpolates(real,fake_patched)
        gp_loss = self.gradients_penalty(interps)
        losses['gp_loss'] = self.WGAN_GP_LAMBDA * gp_loss
        losses['d_loss'] = losses['adv_dloss'] + losses['gp_loss']
        #print('gp loss', losses['gp_loss'])
        loss_D = losses['d_loss']
        return loss_D


class TrainOneStepD(nn.Cell):
    def __init__(self,D,optimizer,sens=1.0):
        super(TrainOneStepD,self).__init__(auto_prefix=True)
        self.optimizer = optimizer
        self.D = D
        self.D.netD.set_grad()
        self.D.netD.set_train()
        self.D.netG.set_grad(False)
        self.D.netG.set_train(False)
        self.grad = ops.GradOperation(get_by_list=True,sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.reducer_flag = False
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL,ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights,mean,degree)

    def construct(self,real,x,mask):
        weights = self.weights
        loss_D = self.D(real,x,mask)
        sens_d = self.fill(self.dtype(loss_D),self.shape(loss_D),self.sens)
        grads_d = self.grad(self.D,weights)(real,x,mask,sens_d)
        if self.reducer_flag:
            grads_d = self.grad_reducer(grads_d)
        self.optimizer(grads_d)
        return loss_D
     

class TrainOneStepG(nn.Cell):
    def __init__(self,G,optimizer,sens=1.0):
        super(TrainOneStepG,self).__init__(auto_prefix=True)
        self.optimizer = optimizer
        self.G = G
        self.G.netG.set_grad()
        self.G.netG.set_train()
        self.G.netD.set_grad(False)
        self.G.netD.set_train(False)
        self.grad = ops.GradOperation(get_by_list=True,sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.reducer_flag = False
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL,ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights,mean,degree)

    def construct(self,real,x,mask):
        weights = self.weights
        loss_G = self.G(real,x,mask)
        sens_g = self.fill(self.dtype(loss_G),self.shape(loss_G),self.sens)
        grads_g = self.grad(self.G,weights)(real,x,mask,sens_g)
        if self.reducer_flag:
            grads_g = self.grad_reducer(grads_g)
        self.optimizer(grads_g)
        return loss_G



def GAN_trainer(args):
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    init('nccl')
    rank_id = get_rank()
    rank_size = get_group_size()
    #print('device',rank_id,rank_size)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,gradients_mean=True)
    dataset_generator = InpaintDataset(args)
    dataset_size = len(dataset_generator)
    dataset = ds.GeneratorDataset(dataset_generator,['image'],num_shards=rank_size, shard_id=rank_id)
    dataset = dataset.batch(args.train_batchsize,drop_remainder=True)
    dataset = dataset.create_dict_iterator()
    netG = GatedGenerator(args) 
    netD = Discriminator(args)
    netG_with_loss = GenWithLossCell(netG,netD,args)
    netD_with_loss = DisWithLossCell(netG,netD,args)
    total_batch = (dataset_size // args.train_batchsize) // rank_size
    lr = nn.exponential_decay_lr(args.learning_rate,args.lr_decrease_factor, total_batch * args.Epochs, total_batch, args.lr_decrease_epoch,True)
    optimizer_G = nn.Adam(filter(lambda p: p.requires_grad,netG.trainable_params()),lr,0.5,0.9)  
    optimizer_D = nn.Adam(netD.trainable_params(),lr,0.5,0.9)
    train_discriminator = TrainOneStepD(netD_with_loss,optimizer_D)
    train_generator = TrainOneStepG(netG_with_loss,optimizer_G)
    train_discriminator.set_train()
    train_generator.set_train()
    for epoch in range(args.Epochs):
        print("Starting Training Loop...")
        for batch_idx,image in enumerate(dataset):
            real = image['image']
            real = real.astype(mindspore.float32)
            mask = random_mask(args)
            x = real * (1-mask)   
            for i in range(args.dis_iter):
                netD_loss = train_discriminator(real, x, mask)
            netG_loss = train_generator(real, x, mask)
            print('epoch{}/{}, batch{}/{}, d_loss is {:.4f}, g_loss is {:.4f}'.format(epoch+1,args.Epochs,batch_idx+1,total_batch,netD_loss.asnumpy(),netG_loss.asnumpy()))
            save_checkpoint_path = args.save_folder + str(get_rank())
            if not os.path.isdir(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)
            gen_name = 'generator_epoch%d_batch%d.ckpt' % (epoch + 1, batch_idx+1)
            dis_name = 'discriminator_epoch%d_batch%d.ckpt' % (epoch + 1, batch_idx+1)
            gen_name = os.path.join(save_checkpoint_path, gen_name) 
            dis_name = os.path.join(save_checkpoint_path, dis_name) 
            if (batch_idx + 1) == total_batch:
                save_checkpoint(train_generator,gen_name)
                save_checkpoint(train_discriminator,dis_name)
            if (epoch+1) == args.Epochs:
                if (batch_idx + 1) % 10000 == 0:
                    save_checkpoint(train_generator,gen_name)
                    save_checkpoint(train_discriminator,dis_name)
     

if __name__ == '__main__':
    GAN_trainer(cra_config)


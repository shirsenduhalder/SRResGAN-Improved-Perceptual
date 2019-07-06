import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
from srresnet import _NetG, _NetD
from dataset import DatasetFromHdf5
from torchvision import models
import torch.utils.model_zoo as model_zoo
from loss import *
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

STEPS = 0
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default="true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
#changed
parser.add_argument("--dis_perceptual_loss", action="store_true", help="Use perceptual loss from discriminator?")
parser.add_argument("--gen_adversarial_loss", action="store_true", help="Use adversarial loss of generator?")
parser.add_argument("--coverage", action="store_true", help="Use coverage?")
parser.add_argument("--sample_dir", default="outputs/samples/", help="Path to save traiing samples")

def main():

    #changed
    # global opt, model, netContent

    global opt, model_G, model_D, netContent, writer, STEPS

    opt = parser.parse_args()
    print(opt)
    writer = SummaryWriter(logdir="outputs/logs/"+'Ploss('+str(opt.dis_perceptual_loss)+')_GANloss('+str(opt.gen_adversarial_loss)+\
                                ')_VGGloss('+str(opt.vgg_loss)+')_coverage('+str(opt.coverage)+')/', comment="-srgan-")

    opt.sample_dir = opt.sample_dir + 'Ploss('+str(opt.dis_perceptual_loss)+')_GANloss('+str(opt.gen_adversarial_loss)+\
                                ')_VGGloss('+str(opt.vgg_loss)+')_coverage('+str(opt.coverage)+')/'

    opt.checkpoint_file = "outputs/checkpoint/" + 'Ploss('+str(opt.dis_perceptual_loss)+')_GANloss('+str(opt.gen_adversarial_loss)+\
                    ')_VGGloss('+str(opt.vgg_loss)+')_coverage('+str(opt.coverage)+')/'


    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = DatasetFromHdf5("data/srresnet_x4.h5")
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
        batch_size=opt.batchSize, shuffle=True)

    print('===> Loading VGG model')
    netVGG = models.vgg19()
    if opt.vgg_loss:
        if os.path.isfile('data/vgg19-dcbb9e9d.pth'):
        	netVGG.load_state_dict(torch.load('data/vgg19-dcbb9e9d.pth'))
        else:
        	netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
    class _content_model(nn.Module):
        def __init__(self):
            super(_content_model, self).__init__()
            self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])
            
        def forward(self, x):
            out = self.feature(x)
            return out

    netContent = _content_model()

    print("===> Building model")
    # changed
    # Building generator and discriminator
    model_G = _NetG()
    
    #changed
    model_D = _NetD()
    criterion_G = GeneratorLoss(netContent, model_D, writer, STEPS)
    criterion_D = nn.BCELoss()

    print("===> Setting GPU")
    if cuda:
        #changed
        model_G = model_G.cuda()
        model_D = model_D.cuda()
        criterion_G = criterion_G.cuda()
        criterion_D = criterion_D.cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda() 

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            # changed
            model_G.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            # changed
            model_G.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    # changed
    optimizer_G = optim.Adam(model_G.parameters(), lr=opt.lr)
    optimizer_D = optim.Adam(model_D.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        # changed
        train(training_data_loader, optimizer_G, optimizer_D, model_G, model_D, criterion_G, criterion_D, epoch, STEPS)
        save_checkpoint(model_G, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def train(training_data_loader, optimizer_G, optimizer_D, model_G, model_D, criterion_G, criterion_D, epoch, STEPS):

    lr = adjust_learning_rate(optimizer_G, epoch-1)
    
    for param_group in optimizer_G.param_groups:
        param_group["lr"] = lr
    
    for param_group in optimizer_D.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer_G.param_groups[0]["lr"]))
    model_G.train()
    model_D.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        

        STEPS += input.shape[0]

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        output = model_G(input)
        
        target_real = Variable(torch.rand(opt.batchSize)*0.5 + 0.7).cuda()
        target_fake = Variable(torch.rand(opt.batchSize)*0.3).cuda()

        model_D.zero_grad()
        real_out = model_D(target)[-1]
        fake_out = model_D(output)[-1]
        loss_d = criterion_D(real_out, target_real) + criterion_D(fake_out, target_fake)
        loss_d.backward(retain_graph=True)
        optimizer_D.step()


        # if opt.vgg_loss:
        #     content_input = netContent(output)
        #     content_target = netContent(target)
        #     content_target = content_target.detach()
        #     content_loss = criterion(content_input, content_target)

        optimizer_G.zero_grad()
        loss_g = criterion_G(opt.gen_adversarial_loss, opt.vgg_loss, opt.dis_perceptual_loss, opt.coverage, fake_out, output, target, opt)

        print(STEPS)

        # if opt.vgg_loss:
        #     netContent.zero_grad()
        #     content_loss.backward(retain_graph=True)

        loss_g.backward()

        optimizer_G.step()

        writer.add_scalar("Loss_G", loss_g.item(), STEPS)
        writer.add_scalar("Loss_D", loss_d.item(), STEPS)

        if iteration%20 == 0:
            # if opt.vgg_loss:
            #     print("===> Epoch[{}]({}/{}): Loss: {:.5} Content_loss {:.5}".format(epoch, iteration, len(training_data_loader), loss.data[0], content_loss.data[0]))
            # else:
            
            sample_img = utils.make_grid(torch.cat([output.detach().clone(), target], dim=0), padding=2, normalize=True)
            if not os.path.exists(opt.sample_dir):
                os.makedirs(opt.sample_dir)
            
            utils.save_image(sample_img, os.path.join(opt.sample_dir, "Epoch-{}--Iteration-{}.png".format(epoch, iteration)), padding=5)

            print("===> Epoch[{}]({}/{}): G_Loss: {:.3}, D_Loss: {:.3} ".format(epoch, iteration, len(training_data_loader), loss_g.item(), loss_d.item()))




def save_checkpoint(model, epoch):
    model_out_path = opt.checkpoint_file + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(opt.checkpoint_file):
        os.makedirs(opt.checkpoint_file)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()

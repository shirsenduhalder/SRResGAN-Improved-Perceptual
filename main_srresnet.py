import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as torch_utils
from srresnet import _NetG, _NetD
from dataset import DatasetFromHdf5
from torchvision import models
import torch.utils.model_zoo as model_zoo
from loss import *
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from dataset_helper import create_dataloader
from dataset_helper import create_dataset
from dataset_helper.common import find_benchmark
import options.options as option
import datetime as dt
from eval_save import *
import pickle

STEPS = 0
BEST_PSNR, BEST_VIF = 0, 0
os.environ["HDF5_USE_FILE_LOCKING"]='FALSE'

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=1000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=5e4, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default="true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--sample_dir", default="outputs/samples/", help="Path to save traiing samples")
parser.add_argument("--logs_dir", default="outputs/logs/", help="Path to save logs")
parser.add_argument("--checkpoint_dir", default="outputs/checkpoint/", help="Path to save checkpoint")
parser.add_argument('-options', default='options/train_SRGAN.json', type=str, help='Path to options JSON file.')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
#changed
parser.add_argument("--target_net_flag", action="store_true", help="Use target network in discriminator for stabler training?")
parser.add_argument("--target_TAU", default=0.001, type=float, help="Mixing ratio for updating the target network")
# parser.add_argument("--target_frequency", default=100, type=int, help="Frequency of updating the target network")
parser.add_argument("--mse_major", action="store_true", help="Set MSE coeff 1 and Percep coeff 0.01")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--adversarial_loss", action="store_true", help="Use adversarial loss of generator?")
parser.add_argument("--dis_perceptual_loss", action="store_true", help="Use perceptual loss from discriminator?")
# parser.add_argument("--huber_loss", action="store_true", help="Uses huber loss for computing perceptual loss from discriminator?")
parser.add_argument("--softmax_loss", action="store_true", help="Use softmax normalized loss for discriminator perceptual loss?")
parser.add_argument("--coverage", action="store_true", help="Use coverage?")
parser.add_argument("--RRDB_block", action="store_true", help="Use content loss?")
#coefficient
parser.add_argument("--mse_loss_coefficient", type=float, default=0.01, help="Coefficient for MSE Loss")
parser.add_argument("--vgg_loss_coefficient", type=float, default=0.5, help="Coefficient for VGG loss")
parser.add_argument("--adversarial_loss_coefficient", type=float, default=0.005, help="Coefficient for adversarial loss")
parser.add_argument("--dis_perceptual_loss_coefficient", type=float, default=1, help="Coefficient for perceptual loss from discriminator")
parser.add_argument("--coverage_coefficient", type=float, default=0.999, help="Mixing ratio / effective horizon")
# parser.add_argument("--dataset", default="DIV2K, Flickr2K", help="Enter Dataset, Default: [DIV2K], Options = ['DIV2K', 'Flickr2K', 'Set5', 'Set14', 'BSD100', 'Sun-Hays80', 'Urban100']")

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def main():

    #changed
    # global opt, model, netContent

    global opt, model_G, model_D, target_model_D, netContent, writer, STEPS

    opt = parser.parse_args()
    if opt.mse_major:
        opt.mse_loss_coefficient = 1
        opt.dis_perceptual_loss_coefficient = 0.01
    opt.huber_loss = True
    options = option.parse(opt.options)
    print(opt)
    # print(options)
    
    if opt.dis_perceptual_loss:
        assert opt.adversarial_loss, "Discriminator perceptual loss is invalid without adversarial loss"
    
    if opt.softmax_loss:
        assert opt.dis_perceptual_loss, "Softmax normalization is only valid for Discriminator Perceptual loss"

    if opt.target_net_flag:
        assert opt.dis_perceptual_loss, "Target network is only valid for Discriminator Perceptual loss"

    if opt.target_net_flag:
    	exp_name = "True_"+str(opt.target_TAU)
    else:
    	exp_name = "False"
    out_folder = "complete_Target({})_mseMajor({})_Softmax({})_PerLoss({})_GANloss({})_VGGloss({})_coverage({})_huber({})_perCoeff({})".format(exp_name, opt.mse_major, opt.softmax_loss, opt.dis_perceptual_loss, opt.adversarial_loss, opt.vgg_loss, opt.coverage, opt.huber_loss, opt.dis_perceptual_loss_coefficient)

    writer = SummaryWriter(logdir = os.path.join(opt.logs_dir, out_folder), comment="-srgan-")

    opt.sample_dir = os.path.join(opt.sample_dir, out_folder)

    opt.checkpoint_file = os.path.join(opt.checkpoint_dir, out_folder)


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
    # train_set = DatasetFromHdf5("data/srresnet_x4.h5")
    # training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
    #     batch_size=opt.batchSize, shuffle=True)

    dataset_opt = options['datasets']['train']
    dataset_opt['batch_size'] = opt.batchSize
    print(dataset_opt)
    train_set = create_dataset(dataset_opt)
    training_data_loader = create_dataloader(train_set, dataset_opt)
    print('===> Train Dataset: %s   Number of images: [%d]' % (train_set.name(), len(train_set)))
    if training_data_loader is None: raise ValueError("[Error] The training data does not exist")
        

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
    model_G = _NetG(opt)
    
    #changed
    if opt.adversarial_loss:
        model_D = _NetD()
        target_model_D = None
        if opt.target_net_flag:
            target_model_D = _NetD()
            hard_update(target_model_D, model_D)
        criterion_D = nn.BCELoss()
    else:
        model_D = None
        target_model_D = None
        criterion_D = None

    criterion_G = GeneratorLoss(netContent, writer, STEPS)

    print("===> Setting GPU")
    if cuda:
        #changed
        model_G = model_G.cuda()

        if opt.adversarial_loss:    
            model_D = model_D.cuda()
            if opt.target_net_flag:
                target_model_D = target_model_D.cuda()
            criterion_D = criterion_D.cuda()

        criterion_G = criterion_G.cuda()
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

    if opt.adversarial_loss:       
        optimizer_D = optim.Adam(model_D.parameters(), lr=opt.lr)
    else:
        optimizer_D = None

    print("===> Pre-fetching validation data for monitoring training")
    test_dump_file = 'data/dump/Test5.pickle'

    if os.path.isfile(test_dump_file):
        with open(test_dump_file, 'rb') as p:
            images_test = pickle.load(p)
        images_hr = images_test['images_hr']
        images_lr = images_test['images_lr']
        print("===>Loading Checkpoint Test images")
    else:
        images_hr, images_lr = create_val_ims()
        print("===>Creating Checkpoint Test images")

    print("===> Training")
    #to track loss type
    opt.losstype_print_tracker = True
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        # changed
        train(training_data_loader, optimizer_G, optimizer_D, model_G, model_D, target_model_D, criterion_G, criterion_D, epoch)
        save_checkpoint(images_hr, images_lr, model_G, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.5 ** (STEPS // opt.step))
    return lr 

def train(training_data_loader, optimizer_G, optimizer_D, model_G, model_D, target_model_D, criterion_G, criterion_D, epoch):

    lr = adjust_learning_rate(optimizer_G, epoch-1)
    
    for param_group in optimizer_G.param_groups:
        param_group["lr"] = lr
    
    if opt.adversarial_loss:
        for param_group in optimizer_D.param_groups:
            param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer_G.param_groups[0]["lr"]))
    model_G.train()

    if opt.adversarial_loss:
        model_D.train()

    start_time = dt.datetime.now()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch['LR']), Variable(batch['HR'], requires_grad=False)
        input = input/255
        target = target/255
        

        STEPS += 1 # input.shape[0]

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        output = model_G(input)
        
        target_real = Variable(torch.rand(input.size()[0])*0.5 + 0.7).cuda()
        target_fake = Variable(torch.rand(input.size()[0])*0.3).cuda()

        target_disc = None
        out_disc = None

        if opt.adversarial_loss:
            model_D.zero_grad()
            target_disc = model_D(target)
            out_disc = model_D(output)
            real_out = target_disc[-1]
            fake_out = out_disc[-1]
            loss_d = criterion_D(real_out, target_real) + criterion_D(fake_out, target_fake)
            loss_d.backward(retain_graph=True)
            optimizer_D.step()
        else:
            fake_out = None

        if opt.target_net_flag:
            target_model_D.eval()
            target_disc = target_model_D(target)
            out_disc = target_model_D(output)
            soft_update(target_model_D, model_D, opt.target_TAU)
            # if STEPS%opt.target_frequency == 0:
            #     hard_update(target_model_D, model_D)


        # if opt.vgg_loss:
        #     content_input = netContent(output)
        #     content_target = netContent(target)
        #     content_target = content_target.detach()
        #     content_loss = criterion(content_input, content_target)

        optimizer_G.zero_grad()
        loss_g = criterion_G(target_disc, out_disc, fake_out, output, target, opt)

        # if opt.vgg_loss:
        #     netContent.zero_grad()
        #     content_loss.backward(retain_graph=True)

        loss_g.backward()

        optimizer_G.step()

        writer.add_scalar("Loss_G", loss_g.item(), STEPS)

        if opt.adversarial_loss:            
            writer.add_scalar("Loss_D", loss_d.item(), STEPS)

        if iteration%1000 == 0:
            # if opt.vgg_loss:
            #     print("===> Epoch[{}]({}/{}): Loss: {:.5} Content_loss {:.5}".format(epoch, iteration, len(training_data_loader), loss.data[0], content_loss.data[0]))
            # else:
            
            sample_img = torch_utils.make_grid(torch.cat([output.detach().clone(), target], dim=0), padding=2, normalize=False)
            if not os.path.exists(opt.sample_dir):
                os.makedirs(opt.sample_dir)
            
            torch_utils.save_image(sample_img, os.path.join(opt.sample_dir, "Epoch-{}--Iteration-{}.png".format(epoch, iteration)), padding=5)

        if iteration%100 == 0:
            if opt.adversarial_loss:
                print("===> Epoch[{}]({}/{}): G_Loss: {:.3}, D_Loss: {:.3}, Time: {} ".format(epoch, iteration, len(training_data_loader), loss_g.item(), loss_d.item(), (dt.datetime.now()-start_time).seconds))
            else:
                print("===> Epoch[{}]({}/{}): Loss: {:.3}, Time: {} ".format(epoch, iteration, len(training_data_loader), loss_g.item(), (dt.datetime.now()-start_time).seconds))
            start_time = dt.datetime.now()

def save_checkpoint(images_hr, images_lr, model, epoch):        

    psnr_test, _, vif_test, _ = eval_metrics(images_hr, images_lr, model, scale_factor=4, cuda=True, show_bicubic=False, save_images=False)

    global BEST_PSNR, BEST_VIF
    
    writer.add_scalar("PSNR", psnr_test, epoch)
    writer.add_scalar("VIF", vif_test, epoch)


    if psnr_test > BEST_PSNR or vif_test > BEST_VIF or epoch%10==0:
        model_out_path = os.path.join(opt.checkpoint_file,  "model_epoch_{}_PSNR_{}_VIF_{}.pth".format(epoch, psnr_test, vif_test))
        state = {"epoch": epoch ,"model": model}
        if not os.path.exists(opt.checkpoint_file):
            os.makedirs(opt.checkpoint_file)

        torch.save(state, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))

        if psnr_test > BEST_PSNR:            
            print("PSNR updated {} ====> {}".format(BEST_PSNR, psnr_test))
            BEST_PSNR = psnr_test
        if vif_test > BEST_VIF:
            print("VIF updated {} ====> {}".format(BEST_VIF, vif_test))
            BEST_VIF = vif_test

def create_val_ims():
    
    images_test = {}
    images_hr, images_lr = [], []

    test_root = 'data/test/Set5/image_SRF_4'
    img_hr_names = natsorted(glob2.glob(test_root + "/*HR*.*"))
    img_lr_names = natsorted(glob2.glob(test_root + "/*LR*.*"))

    for hr, lr in zip(img_hr_names, img_lr_names):
        images_hr.append(sio.imread(hr))
        images_lr.append(sio.imread(lr))
    images_test['images_hr'] = images_hr
    images_test['images_lr'] = images_lr

    dump_dir = "data/dump/"

    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    
    with open(os.path.join(dump_dir, 'Test5.pickle'), 'wb') as p:
        pickle.dump(images_test, p, protocol=pickle.HIGHEST_PROTOCOL)
    
    return images_hr, images_lr

if __name__ == "__main__":
    main()
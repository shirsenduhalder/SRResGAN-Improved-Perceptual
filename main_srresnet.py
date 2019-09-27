import argparse, os
from copy import deepcopy
from basic_utils import Fvp, conjugate_gradients
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
parser.add_argument("--max_updates", type=int, default=1e6, help="number of updates during training")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument('-options', default='options/train_SRGAN.json', type=str, help='Path to options JSON file.')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--pretrained", default="model/MSE_major_1500.pth", type=str, help="path to pretrained model (default: initialized with MSE major model)")
parser.add_argument("--lr_disc", type=float, default=1e-4, help="Learning Rate. Default=1e-4")

parser.add_argument("--sample_dir", default="outputs/samples/", help="Path to save traiing samples")
parser.add_argument("--fine_sample_dir", default="outputs/finetune_samples/", help="Path to save traiing samples")
parser.add_argument("--logs_dir", default="outputs/logs/", help="Path to save logs")
parser.add_argument("--checkpoint_dir", default="outputs/checkpoint/", help="Path to save checkpoint")
parser.add_argument("--fine_checkpoint_dir", default="outputs/finetune_checkpoint/", help="Path to save checkpoint")
parser.add_argument("--epoch_frequency", type=int, default=50, help="NA")
parser.add_argument("--epoch_finetune_frequency", type=int, default=10, help="NA")

parser.add_argument("--CG_steps", type=int, default=20, help="Number of CG steps to compute the meta-gradient")
#coefficient
parser.add_argument("--mse_loss_coefficient_inner", type=float, default=0.01, help="Coefficient for MSE Loss in the inner loop")
parser.add_argument("--mse_loss_coefficient_outer", type=float, default=0.01, help="Coefficient for MSE Loss in the outer loop")
parser.add_argument("--vgg_loss_coefficient", type=float, default=0.5, help="Coefficient for VGG loss")
parser.add_argument("--adversarial_loss_coefficient", type=float, default=0.005, help="Coefficient for adversarial loss")
parser.add_argument("--preservation_loss_coefficient", type=float, default=0.1, help="Coefficient for preservation loss")
parser.add_argument("--inner_loop_steps", type=int, default=5, help="number of steps to be taken in the inner loop")
parser.add_argument("--lr_inner", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr_outer", type=float, default=1e-4, help="Learning Rate. Default=1e-4")

def main():

    global opt, model_G, model_D, netContent, writer, STEPS

    opt = parser.parse_args()
    options = option.parse(opt.options)
    print(opt)

    out_folder = "steps({})_lrIN({})_lrOUT({})_lambda(mseIN={},mseOUT={},vgg={},adv={},preserve={})".format(
    	opt.inner_loop_steps, opt.lr_inner, opt.lr_outer, opt.mse_loss_coefficient_inner, opt.mse_loss_coefficient_outer,
    	opt.vgg_loss_coefficient, opt.adversarial_loss_coefficient, opt.preservation_loss_coefficient)

    writer = SummaryWriter(logdir = os.path.join(opt.logs_dir, out_folder), comment="-srgan-")

    opt.sample_dir = os.path.join(opt.sample_dir, out_folder)
    opt.fine_sample_dir = os.path.join(opt.fine_sample_dir, out_folder)

    opt.checkpoint_file_init = os.path.join(opt.checkpoint_dir, "init/"+out_folder)
    opt.checkpoint_file_final = os.path.join(opt.checkpoint_dir, "final/"+out_folder)
    opt.checkpoint_file_fine = os.path.join(opt.fine_checkpoint_dir, out_folder)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    dataset_opt = options['datasets']['train']
    dataset_opt['batch_size'] = opt.batchSize
    print(dataset_opt)
    train_set = create_dataset(dataset_opt)
    training_data_loader = create_dataloader(train_set, dataset_opt)
    print('===> Train Dataset: %s   Number of images: [%d]' % (train_set.name(), len(train_set)))
    if training_data_loader is None: raise ValueError("[Error] The training data does not exist")

    print('===> Loading VGG model')
    netVGG = models.vgg19()
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

    G_init = _NetG(opt).cuda()
    model_D = _NetD().cuda()
    netContent = _content_model().cuda()
    criterion_G = GeneratorLoss(netContent, writer, STEPS).cuda()
    criterion_D = nn.BCELoss().cuda()

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        assert os.path.isfile(opt.pretrained)
        print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)
        # changed
        G_init.load_state_dict(weights['model'].state_dict())


    print("===> Setting Optimizer")
    # changed
    optimizer_G_outer = optim.Adam(G_init.parameters(), lr=opt.lr_outer)
    optimizer_D = optim.Adam(model_D.parameters(), lr=opt.lr_disc)

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
    epoch = opt.start_epoch
    try:
        while STEPS < (opt.inner_loop_steps+1)*opt.max_updates:
            # changed
            last_model_G = train(training_data_loader, optimizer_G_outer, optimizer_D, G_init, model_D, criterion_G, criterion_D, epoch, STEPS, writer)
            assert last_model_G is not None
            save_checkpoint(images_hr, images_lr, G_init, last_model_G, epoch)
            epoch += 1
    except KeyboardInterrupt:
        print("KeyboardInterrupt HANDLED! Running the final epoch on G_init")
    epoch += 1
    if STEPS < 5e4:
        lr_finetune = opt.lr_inner
    elif STEPS < 1e5:
        lr_finetune = opt.lr_inner/2
    elif STEPS < 2e5:
        lr_finetune = opt.lr_inner/4
    elif STEPS < 4e5:
        lr_finetune =  opt.lr_inner/8
    elif STEPS < 8e5:
        lr_finetune = opt.lr_inner/16
    else:
        lr_finetune = opt.lr_inner/32

    model_G = deepcopy(G_init)
    optimizer_G_inner = optim.Adam(model_G.parameters(), lr=lr_finetune)
    model_G.train()
    optimizer_G_inner.zero_grad()
    init_parameters = torch.cat([p.view(-1) for k, p in G_init.named_parameters() if p.requires_grad])

    opt.adversarial_loss = False
    opt.vgg_loss = True
    opt.mse_loss_coefficient = opt.mse_loss_coefficient_inner

    start_time = dt.datetime.now()
    total_num_examples = len(training_data_loader)
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch['LR']), Variable(batch['HR'], requires_grad=False)
        input = input.cuda()/255
        target = target.cuda()/255
        STEPS += 1
        output = model_G(input)
        fake_out = None
        optimizer_G_inner.zero_grad()
        loss_g_inner = criterion_G(fake_out, output, target, opt)
        curr_parameters = torch.cat([p.view(-1) for k, p in model_G.named_parameters() if p.requires_grad])
        preservation_loss = ((Variable(init_parameters).detach()-curr_parameters)**2).sum()
        loss_g_inner += preservation_loss
        loss_g_inner.backward()
        optimizer_G_inner.step()
        writer.add_scalar("Loss_G_finetune", loss_g_inner.item(), STEPS)
        if iteration%5 == 0:
            fine_sample_img = torch_utils.make_grid(torch.cat([output.detach().clone(), target], dim=0), padding=2, normalize=False)
            if not os.path.exists(opt.fine_sample_dir):
                os.makedirs(opt.fine_sample_dir)
            torch_utils.save_image(fine_sample_img, os.path.join(opt.fine_sample_dir, "Epoch-{}--Iteration-{}.png".format(epoch, iteration)), padding=5)

            print("===> Finetuning Epoch[{}]({}/{}): G_Loss(finetune): {:.3}".format(epoch, iteration, total_num_examples,
            	loss_g_inner.item(), (dt.datetime.now()-start_time).seconds))
            start_time = dt.datetime.now()
            save_checkpoint(images_hr, images_lr, None, model_G, iteration, finetune = True)
    save_checkpoint(images_hr, images_lr, None, model_G, total_num_examples, finetune = True)


def train(training_data_loader, optimizer_G_outer, optimizer_D, G_init, model_D, criterion_G, criterion_D, epoch, STEPS, writer):

    if STEPS < 5e4:
        lr = opt.lr_inner
    elif STEPS < 1e5:
        lr = opt.lr_inner/2
    elif STEPS < 2e5:
        lr = opt.lr_inner/4
    elif STEPS < 4e5:
        lr =  opt.lr_inner/8
    elif STEPS < 8e5:
        lr = opt.lr_inner/16
    else:
        lr = opt.lr_inner/32

    model_G = deepcopy(G_init)
    optimizer_G_inner = optim.Adam(model_G.parameters(), lr=lr)
    G_init.train()
    model_G.train()
    model_D.train()
    optimizer_G_inner.zero_grad()
    optimizer_G_outer.zero_grad()
    optimizer_D.zero_grad()
    init_parameters = torch.cat([p.view(-1) for k, p in G_init.named_parameters() if p.requires_grad])

    start_time = dt.datetime.now()
    total_num_examples = len(training_data_loader)
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch['LR']), Variable(batch['HR'], requires_grad=False)
        input = input.cuda()/255
        target = target.cuda()/255
        STEPS += 1

        opt.adversarial_loss = False
        opt.vgg_loss = True
        opt.mse_loss_coefficient = opt.mse_loss_coefficient_inner


        output = model_G(input)
        if iteration%(opt.inner_loop_steps+1) == 0:
            model_D.zero_grad()
            real_out = model_D(target)
            fake_out = model_D(output)
            target_real = Variable(torch.rand(input.size()[0])*0.5 + 0.7).cuda()
            target_fake = Variable(torch.rand(input.size()[0])*0.3).cuda()
            loss_d = criterion_D(real_out, target_real) + criterion_D(fake_out, target_fake)
            loss_d.backward(retain_graph=True)
            optimizer_D.step()
        else:
            fake_out = None

        optimizer_G_inner.zero_grad()
        loss_g_inner = criterion_G(fake_out, output, target, opt)
        curr_parameters = torch.cat([p.view(-1) for k, p in model_G.named_parameters() if p.requires_grad])
        preservation_loss = ((Variable(init_parameters).detach()-curr_parameters)**2).sum()
        loss_g_inner += preservation_loss
        loss_g_inner.backward(retain_graph=True)
        if iteration%(opt.inner_loop_steps+1) == 0:
            inner_grads = []
            for model_param in model_G.parameters():
                inner_grads.append(model_param.grad.view(-1))
            inner_grads = torch.cat(inner_grads)
            optimizer_G_inner.zero_grad()
            # ----------------------------------------------------------------------------------------------------------
            opt.adversarial_loss = True
            opt.vgg_loss = False
            opt.mse_loss_coefficient = opt.mse_loss_coefficient_outer
            output_new = model_G(input)
            loss_g_outer = criterion_G(fake_out, output_new, target, opt)
            loss_g_outer.backward()
            outer_grads = []
            for param in model_G.parameters():
                outer_grads.append(param.grad.view(-1))
            outer_grads = torch.cat(outer_grads)
            optimizer_G_inner.zero_grad()
            # ----------------------------------------------------------------------------------------------------------
            implicit_grad = conjugate_gradients(Fvp, inner_grads, outer_grads, model_G, optimizer_G_inner, 50, meta_lambda=opt.preservation_loss_coefficient, cuda = True)
            optimizer_G_inner.zero_grad()
            optimizer_G_outer.zero_grad()
            prev_ind = 0
            for param in G_init.parameters():
                flat_size = int(np.prod(list(param.size())))
                param.grad = implicit_grad[prev_ind:prev_ind + flat_size].view(param.size())
                prev_ind += flat_size
            optimizer_G_outer.step()
        else:
            optimizer_G_inner.step()

        writer.add_scalar("Loss_G_inner", loss_g_inner.item(), STEPS)
        if iteration%(opt.inner_loop_steps+1) == 0:
            writer.add_scalar("Loss_G_outer", loss_g_outer.item(), STEPS)
            writer.add_scalar("Loss_D", loss_d.item(), STEPS)
            if iteration >= total_num_examples-opt.inner_loop_steps:
                sample_img = torch_utils.make_grid(torch.cat([output.detach().clone(), target], dim=0), padding=2, normalize=False)
                if not os.path.exists(opt.sample_dir):
                    os.makedirs(opt.sample_dir)            
                torch_utils.save_image(sample_img, os.path.join(opt.sample_dir, "Epoch-{}--Iteration-{}.png".format(epoch, iteration)), padding=5)

        if iteration%(opt.inner_loop_steps+1) == 0: # and (iteration//(opt.inner_loop_steps+1))%(100//(opt.inner_loop_steps+1)) == 0:
            print("===> Epoch[{}]({}/{}): G_Loss: {:.3} (inner)/ {:.3} (outer), D_Loss: {:.3}, Time: {} ".format(epoch, iteration, total_num_examples,
            	loss_g_inner.item(), loss_g_outer.item(), loss_d.item(), (dt.datetime.now()-start_time).seconds))
            start_time = dt.datetime.now()
            if iteration >= total_num_examples-opt.inner_loop_steps:
                return model_G
            else:
                model_G = deepcopy(G_init)
                optimizer_G_inner = optim.Adam(model_G.parameters(), lr=lr)
                optimizer_G_inner.zero_grad()
                optimizer_G_outer.zero_grad()
                optimizer_D.zero_grad()
                init_parameters = torch.cat([p.view(-1) for k, p in G_init.named_parameters() if p.requires_grad])

def save_checkpoint(images_hr, images_lr, model_init, model_k_steps, epoch, finetune = False):        

    psnr_test, _, vif_test, _ = eval_metrics(images_hr, images_lr, model_k_steps, scale_factor=4, cuda=True, show_bicubic=False, save_images=False)

    global BEST_PSNR, BEST_VIF
    
    if not finetune:
        writer.add_scalar("PSNR", psnr_test, epoch)
        writer.add_scalar("VIF", vif_test, epoch)

    frequency = opt.epoch_finetune_frequency if finetune else opt.epoch_frequency
    if psnr_test > BEST_PSNR or vif_test > BEST_VIF or epoch%frequency==0:
        if not finetune:
            model_out_path_init = os.path.join(opt.checkpoint_file_init,  "model_epoch_{}_PSNR_{}_VIF_{}.pth".format(epoch, psnr_test, vif_test))
            state_init = {"epoch": epoch ,"model": model_init}
            if not os.path.exists(opt.checkpoint_file_init):
                os.makedirs(opt.checkpoint_file_init)
            state_init = {"epoch": epoch ,"model": model_init}
            torch.save(state_init, model_out_path_init)

            model_out_path_final = os.path.join(opt.checkpoint_file_final,  "model_epoch_{}_PSNR_{}_VIF_{}.pth".format(epoch, psnr_test, vif_test))
            if not os.path.exists(opt.checkpoint_file_final):
                os.makedirs(opt.checkpoint_file_final)
            state_final = {"epoch": epoch ,"model": model_k_steps}
        else:
            model_out_path_final = os.path.join(opt.checkpoint_file_fine,  "model_iteration_{}_PSNR_{}_VIF_{}.pth".format(epoch, psnr_test, vif_test))
            if not os.path.exists(opt.checkpoint_file_fine):
                os.makedirs(opt.checkpoint_file_fine)
            model_out_path_init = "<NiL>"
            state_final = {"iteration": epoch ,"model": model_k_steps}
        torch.save(state_final, model_out_path_final)

        print("Checkpoint saved to {}, {}".format(model_out_path_init, model_out_path_final))

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
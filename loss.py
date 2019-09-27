import torch
import torch.nn as nn
from torch.autograd import Variable

class GeneratorLoss(nn.Module):
    def __init__(self, vgg_network, writer, steps):
        super(GeneratorLoss, self).__init__()
        self.vgg_network = vgg_network
        # self.dis_network = dis_network
        self.writer = writer
        self.steps = steps
        self.mse_loss = nn.MSELoss().cuda()
        self.bce_loss = nn.BCELoss().cuda()
        self.huber_loss = nn.SmoothL1Loss().cuda()
    
    def forward(self, out_labels, out_images, target_images, opt):
        # self.steps += out_images.shape[0]
        # print("Image loss: {}".format(image_loss.item()))

        overall_loss = 0
        self.ones_const = Variable(torch.ones(out_images.size()[0])).cuda()

        image_loss = self.huber_loss(out_images, target_images)
        self.writer.add_scalar("Image Loss", image_loss, self.steps)
        overall_loss += opt.mse_loss_coefficient * image_loss

        if opt.adversarial_loss:
            adversarial_loss = self.bce_loss(out_labels, self.ones_const)
            self.writer.add_scalar("Gen Adversarial Loss", adversarial_loss, self.steps)
            overall_loss += opt.adversarial_loss_coefficient*adversarial_loss

        if opt.vgg_loss:
            vgg_perception_loss = self.mse_loss(self.vgg_network(out_images), self.vgg_network(target_images))
            self.writer.add_scalar("VGG Perception Loss", vgg_perception_loss, self.steps)
            overall_loss += opt.vgg_loss_coefficient*vgg_perception_loss

        return overall_loss
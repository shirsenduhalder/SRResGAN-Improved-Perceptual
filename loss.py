import torch
import torch.nn as nn
from torch.autograd import Variable

class GeneratorLoss(nn.Module):
    def __init__(self, vgg_network, dis_network):
        super(GeneratorLoss, self).__init__()
        self.vgg_network = vgg_network
        self.dis_network = dis_network
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, gen_adversarial_loss, vgg_loss, dis_perceptual_loss, coverage, out_labels, out_images, target_images, opt):
        image_loss = self.mse_loss(out_images, target_images)
        # print("Image loss: {}".format(image_loss.item()))

        self.ones_const = Variable(torch.ones(opt.batchSize))
        if opt.cuda:
            self.ones_const = self.ones_const.cuda()
            self.mse_loss = self.mse_loss.cuda()
            self.bce_loss = self.bce_loss.cuda()

        if gen_adversarial_loss:    
            adversarial_loss = self.bce_loss(out_labels, self.ones_const)
            # print("Adversarial Loss: {}".format(adversarial_loss.item()))

        if vgg_loss:
            perception_loss = self.mse_loss(self.vgg_network(out_images), self.vgg_network(target_images))
            # print("VGG Loss: {}".format(perception_loss.item()))
        
        if dis_perceptual_loss:
            coverage = 1

            target_1, target_2, target_3, target_4, target_5, target_6, target_7 = self.dis_network(target_images)[:-1]

            out_1, out_2, out_3, out_4, out_5, out_6, out_7 = self.dis_network(out_images)[:-1]

            loss0 = self.mse_loss(target_images, out_images)
            loss1 = self.mse_loss(target_1, out_1)
            loss2 = self.mse_loss(target_2, out_2)
            loss3 = self.mse_loss(target_3, out_3)
            loss4 = self.mse_loss(target_4, out_4)
            loss5 = self.mse_loss(target_5, out_5)
            loss6 = self.mse_loss(target_6, out_6)
            loss7 = self.mse_loss(target_7, out_7)

            sum_exp_loss = torch.exp(loss0) + torch.exp(loss1) + torch.exp(loss2) + torch.exp(loss3) + torch.exp(loss4) + torch.exp(loss5) + torch.exp(loss6) + torch.exp(loss7)

            softmax_loss0 = torch.exp(loss0)/sum_exp_loss
            softmax_loss1 = torch.exp(loss1)/sum_exp_loss
            softmax_loss2 = torch.exp(loss2)/sum_exp_loss
            softmax_loss3 = torch.exp(loss3)/sum_exp_loss
            softmax_loss4 = torch.exp(loss4)/sum_exp_loss
            softmax_loss5 = torch.exp(loss5)/sum_exp_loss
            softmax_loss6 = torch.exp(loss6)/sum_exp_loss
            softmax_loss7 = torch.exp(loss7)/sum_exp_loss

            if coverage:
                coverage0 = 0.9 * coverage + 0.1 * softmax_loss0
                coverage1 = 0.9 * coverage + 0.1 * softmax_loss1    
                coverage2 = 0.9 * coverage + 0.1 * softmax_loss2
                coverage3 = 0.9 * coverage + 0.1 * softmax_loss3
                coverage4 = 0.9 * coverage + 0.1 * softmax_loss4
                coverage5 = 0.9 * coverage + 0.1 * softmax_loss5
                coverage6 = 0.9 * coverage + 0.1 * softmax_loss6
                coverage7 = 0.9 * coverage + 0.1 * softmax_loss7

            perception_loss = (loss0 * softmax_loss0/coverage0) + (loss1 * softmax_loss1/coverage1) + (loss2 * softmax_loss2/coverage2) + (loss3 * softmax_loss3/coverage3) + (loss4 * softmax_loss4/coverage4) + (loss5 * softmax_loss5/coverage5) + (loss6 * softmax_loss6/coverage6) + (loss7 * softmax_loss7/coverage7)

            # print("Perception Loss: {}".format(perception_loss.item()))

        if gen_adversarial_loss and (vgg_loss or dis_perceptual_loss):
            return image_loss + adversarial_loss + perception_loss
        
        elif (vgg_loss or dis_perceptual_loss):
            return image_loss + perception_loss
        
        elif gen_adversarial_loss and not (vgg_loss or dis_perceptual_loss):
            return image_loss + adversarial_loss
        
        else:
            return image_loss
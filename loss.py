import torch
import torch.nn as nn
from torch.autograd import Variable

class GeneratorLoss(nn.Module):
    def __init__(self, vgg_network, dis_network, writer, steps):
        super(GeneratorLoss, self).__init__()
        self.vgg_network = vgg_network
        self.dis_network = dis_network
        self.writer = writer
        self.steps = steps
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.huber_loss = nn.SmoothL1Loss()
    
    def forward(self, out_labels, out_images, target_images, opt):
        self.steps += out_images.shape[0]
        # print("Image loss: {}".format(image_loss.item()))

        self.ones_const = Variable(torch.ones(out_images.size()[0]))
        if opt.cuda:
            self.ones_const = self.ones_const.cuda()
            self.mse_loss = self.mse_loss.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.huber_loss = self.huber_loss.cuda()

        self.hybrid_l2_l1_loss = self.mse_loss
        if opt.huber_loss:
            self.hybrid_l2_l1_loss = self.huber_loss

        image_loss = self.mse_loss(out_images, target_images)
        self.writer.add_scalar("Image Loss", image_loss, self.steps)

        if opt.adversarial_loss:
            adversarial_loss = self.bce_loss(out_labels, self.ones_const)
            self.writer.add_scalar("Gen Adversarial Loss", adversarial_loss, self.steps)
            # print("Adversarial Loss: {}".format(adversarial_loss.item()))

        if opt.vgg_loss:
            vgg_perception_loss = self.mse_loss(self.vgg_network(out_images), self.vgg_network(target_images))
            self.writer.add_scalar("VGG Perception Loss", vgg_perception_loss, self.steps)
            # print("VGG Loss: {}".format(perception_loss.item()))
        
        if opt.dis_perceptual_loss and opt.adversarial_loss:
            coverage0, coverage1, coverage2, coverage3, coverage4, coverage5, coverage6, coverage7 = 1, 1, 1, 1, 1, 1, 1, 1
            softmax_loss0, softmax_loss1, softmax_loss2, softmax_loss3, softmax_loss4, softmax_loss5, softmax_loss6, softmax_loss7 = 1, 1, 1, 1, 1, 1, 1, 1

            target_1, target_2, target_3, target_4, target_5, target_6, target_7 = self.dis_network(target_images)[:-1]

            out_1, out_2, out_3, out_4, out_5, out_6, out_7 = self.dis_network(out_images)[:-1]

            loss0 = self.hybrid_l2_l1_loss(target_images, out_images)
            loss1 = self.hybrid_l2_l1_loss(target_1, out_1)
            loss2 = self.hybrid_l2_l1_loss(target_2, out_2)
            loss3 = self.hybrid_l2_l1_loss(target_3, out_3)
            loss4 = self.hybrid_l2_l1_loss(target_4, out_4)
            loss5 = self.hybrid_l2_l1_loss(target_5, out_5)
            loss6 = self.hybrid_l2_l1_loss(target_6, out_6)
            loss7 = self.hybrid_l2_l1_loss(target_7, out_7)

            if opt.softmax_loss:
                sum_exp_loss = torch.exp(loss0) + torch.exp(loss1) + torch.exp(loss2) + torch.exp(loss3) + torch.exp(loss4) + torch.exp(loss5) + torch.exp(loss6) + torch.exp(loss7)

                softmax_loss0 = torch.exp(loss0)/sum_exp_loss
                softmax_loss1 = torch.exp(loss1)/sum_exp_loss
                softmax_loss2 = torch.exp(loss2)/sum_exp_loss
                softmax_loss3 = torch.exp(loss3)/sum_exp_loss
                softmax_loss4 = torch.exp(loss4)/sum_exp_loss
                softmax_loss5 = torch.exp(loss5)/sum_exp_loss
                softmax_loss6 = torch.exp(loss6)/sum_exp_loss
                softmax_loss7 = torch.exp(loss7)/sum_exp_loss

                self.writer.add_scalar("Dis Perceptual Softmax Loss-0", softmax_loss0, self.steps)
                self.writer.add_scalar("Dis Perceptual Softmax Loss-1", softmax_loss1, self.steps)
                self.writer.add_scalar("Dis Perceptual Softmax Loss-2", softmax_loss2, self.steps)
                self.writer.add_scalar("Dis Perceptual Softmax Loss-3", softmax_loss3, self.steps)
                self.writer.add_scalar("Dis Perceptual Softmax Loss-4", softmax_loss4, self.steps)
                self.writer.add_scalar("Dis Perceptual Softmax Loss-5", softmax_loss5, self.steps)
                self.writer.add_scalar("Dis Perceptual Softmax Loss-6", softmax_loss6, self.steps)
                self.writer.add_scalar("Dis Perceptual Softmax Loss-7", softmax_loss7, self.steps)

                self.writer.add_scalar("Dis Sum Softmax Loss", sum_exp_loss, self.steps)

            if opt.coverage:
                coverage0 = (opt.coverage_coefficient) * coverage0 + (1 - opt.coverage_coefficient) * softmax_loss0
                coverage1 = (opt.coverage_coefficient) * coverage1 + (1 - opt.coverage_coefficient) * softmax_loss1
                coverage2 = (opt.coverage_coefficient) * coverage2 + (1 - opt.coverage_coefficient) * softmax_loss2
                coverage3 = (opt.coverage_coefficient) * coverage3 + (1 - opt.coverage_coefficient) * softmax_loss3
                coverage4 = (opt.coverage_coefficient) * coverage4 + (1 - opt.coverage_coefficient) * softmax_loss4
                coverage5 = (opt.coverage_coefficient) * coverage5 + (1 - opt.coverage_coefficient) * softmax_loss5
                coverage6 = (opt.coverage_coefficient) * coverage6 + (1 - opt.coverage_coefficient) * softmax_loss6
                coverage7 = (opt.coverage_coefficient) * coverage7 + (1 - opt.coverage_coefficient) * softmax_loss7

                self.writer.add_scalar("Dis coverage-0", coverage0, self.steps)
                self.writer.add_scalar("Dis coverage-1", coverage1, self.steps)
                self.writer.add_scalar("Dis coverage-2", coverage2, self.steps)
                self.writer.add_scalar("Dis coverage-3", coverage3, self.steps)
                self.writer.add_scalar("Dis coverage-4", coverage4, self.steps)
                self.writer.add_scalar("Dis coverage-5", coverage5, self.steps)
                self.writer.add_scalar("Dis coverage-6", coverage6, self.steps)
                self.writer.add_scalar("Dis coverage-7", coverage7, self.steps)

            dis_perception_loss = (loss0 * softmax_loss0/coverage0) + (loss1 * softmax_loss1/coverage1) + (loss2 * softmax_loss2/coverage2) + (loss3 * softmax_loss3/coverage3) + (loss4 * softmax_loss4/coverage4) + (loss5 * softmax_loss5/coverage5) + (loss6 * softmax_loss6/coverage6) + (loss7 * softmax_loss7/coverage7)

            self.writer.add_scalar("Dis Perceptual Loss-0", loss0, self.steps)
            self.writer.add_scalar("Dis Perceptual Adj Loss-0", (loss0 * softmax_loss0/coverage0), self.steps)

            self.writer.add_scalar("Dis Perceptual Loss-1", loss1, self.steps)
            self.writer.add_scalar("Dis Perceptual Adj Loss-1", (loss1 * softmax_loss1/coverage1), self.steps)

            self.writer.add_scalar("Dis Perceptual Loss-2", loss2, self.steps)
            self.writer.add_scalar("Dis Perceptual Adj Loss-2", (loss2 * softmax_loss2/coverage2), self.steps)

            self.writer.add_scalar("Dis Perceptual Loss-3", loss3, self.steps)
            self.writer.add_scalar("Dis Perceptual Adj Loss-3", (loss3 * softmax_loss3/coverage3), self.steps)
            
            self.writer.add_scalar("Dis Perceptual Loss-4", loss4, self.steps)
            self.writer.add_scalar("Dis Perceptual Adj Loss-4", (loss4 * softmax_loss4/coverage4), self.steps)

            self.writer.add_scalar("Dis Perceptual Loss-5", loss5, self.steps)
            self.writer.add_scalar("Dis Perceptual Adj Loss-5", (loss5 * softmax_loss5/coverage5), self.steps)

            self.writer.add_scalar("Dis Perceptual Loss-6", loss6, self.steps)
            self.writer.add_scalar("Dis Perceptual Adj Loss-6", (loss6 * softmax_loss6/coverage6), self.steps)

            self.writer.add_scalar("Dis Perceptual Loss-7", loss7, self.steps)
            self.writer.add_scalar("Dis Perceptual Adj Loss-7", (loss7 * softmax_loss7/coverage7), self.steps)

            self.writer.add_scalar("Dis Perceptual Loss", dis_perception_loss, self.steps)
            
            # print("Perception Loss: {}".format(perception_loss.item()))

        if opt.adversarial_loss and not opt.vgg_loss and not opt.dis_perceptual_loss:
            if opt.losstype_print_tracker:
                print("+"*30)
                print("IMAGE LOSS + ADVERSARIAL LOSS")
                print("+"*30)
                opt.losstype_print_tracker = False
            return opt.mse_loss_coefficient * image_loss + opt.adversarial_loss_coefficient*adversarial_loss
        
        elif opt.adversarial_loss and opt.vgg_loss and not opt.dis_perceptual_loss:
            if opt.losstype_print_tracker:
                print("+"*30)
                print("IMAGE LOSS + ADVERSARIAL LOSS + VGG LOSS")
                print("+"*30)
                opt.losstype_print_tracker = False
            return opt.mse_loss_coefficient * image_loss + opt.adversarial_loss_coefficient*adversarial_loss + opt.vgg_loss_coefficient*vgg_perception_loss
        
        elif not opt.adversarial_loss and opt.vgg_loss:
            if opt.losstype_print_tracker:
                print("+"*30)
                print("IMAGE LOSS + VGG LOSS")
                print("+"*30)
                opt.losstype_print_tracker = False
            return opt.mse_loss_coefficient * image_loss + opt.vgg_loss_coefficient*vgg_perception_loss
        
        elif opt.dis_perceptual_loss and opt.adversarial_loss and opt.coverage and not opt.vgg_loss:
            if opt.losstype_print_tracker:
                print("+"*30)
                print("IMAGE LOSS + ADVERSARIAL LOSS + DIS PERCEPTUAL LOSS - COVERAGE")
                print("+"*30)
                opt.losstype_print_tracker = False
            return opt.mse_loss_coefficient * image_loss + opt.adversarial_loss_coefficient*adversarial_loss + opt.dis_perceptual_loss_coefficient*dis_perception_loss
        
        elif opt.dis_perceptual_loss and opt.adversarial_loss and not opt.vgg_loss:
            if opt.losstype_print_tracker:
                print("+"*30)
                if opt.softmax_loss:
                    print("IMAGE LOSS + ADVERSARIAL LOSS + DIS PERCEPTUAL LOSS - NO COVERAGE SOFTMAX LOSS")
                else:
                    print("IMAGE LOSS + ADVERSARIAL LOSS + DIS PERCEPTUAL LOSS - NO COVERAGE")
                print("+"*30)
                opt.losstype_print_tracker = False
            return opt.mse_loss_coefficient * image_loss + opt.adversarial_loss_coefficient*adversarial_loss + opt.dis_perceptual_loss_coefficient*dis_perception_loss
        
        elif opt.dis_perceptual_loss and opt.adversarial_loss and opt.vgg_loss and opt.coverage:
            if opt.losstype_print_tracker:
                print("+"*30)
                print("IMAGE LOSS + ADVERSARIAL LOSS + DIS PERCEPTUAL LOSS - COVERAGE + VGG LOSS")
                print("+"*30)
                opt.losstype_print_tracker = False
            return opt.mse_loss_coefficient * image_loss + opt.adversarial_loss_coefficient*adversarial_loss + opt.dis_perceptual_loss_coefficient*dis_perception_loss + opt.vgg_loss_coefficient*vgg_perception_loss
        
        elif opt.dis_perceptual_loss and opt.adversarial_loss and opt.vgg_loss:
            if opt.losstype_print_tracker:
                print("+"*30)
                print("IMAGE LOSS + ADVERSARIAL LOSS + DIS PERCEPTUAL LOSS - NO COVERAGE + VGG LOSS")
                print("+"*30)
                opt.losstype_print_tracker = False
            return opt.mse_loss_coefficient * image_loss + opt.adversarial_loss_coefficient*adversarial_loss + opt.dis_perceptual_loss_coefficient*dis_perception_loss + opt.vgg_loss_coefficient*vgg_perception_loss
        else:
            if opt.losstype_print_tracker:
                print("+"*30)
                print("IMAGE LOSS")
                print("+"*30)
                opt.losstype_print_tracker = False
            return image_loss
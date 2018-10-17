#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchwordemb
import torchvision.models as models
import pdb


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss()
        self.register_buffer('target', torch.tensor(0.0))

    def get_target_tensor(self, input):
        target_tensor = self.target

        return target_tensor.expand_as(input)

    def __call__(self, input):
        target_tensor = self.get_target_tensor(input)
        return self.loss(input, target_tensor)


class GEN(nn.Module):
    """docstring for GEN"""
    def __init__(self, G_w1, G_w2, G_b1, G_b2, temperature):
        super(GEN, self).__init__()
        self.G_w1 = G_w1
        self.G_w2 = G_w2
        self.G_b1 = G_b1
        self.G_b2 = G_b2
        self.temperature = temperature
        # self.softmax = torch.nn.Softmax()

    def pred_score(self, pred_data):
        self.score = (torch.mm(torch.tanh(torch.mm(pred_data.float(), self.G_w1.float()) + self.G_b1), self.G_w2) + self.G_b2) / self.temperature
        return self.score

    def forward(self, pred_data, sample_index, reward, important_sampling):
        softmax_score = F.softmax(self.pred_score(pred_data).view(1, -1), -1)
        gan_prob = torch.gather(softmax_score.view(-1), 0, sample_index.long()).clamp(min=1e-8)
        loss = -torch.mean(torch.log(gan_prob) * reward.view(-1) * important_sampling.view(-1))
        # if torch.isnan(loss):
        #     pdb.set_trace()
        # loss = -torch.mean(torch.log(gan_prob) * reward * important_sampling)
        return loss


class DIS(nn.Module):
    def __init__(self, D_w1, D_w2, D_b1, D_b2):
        super(DIS, self).__init__()
        self.D_w1 = D_w1
        self.D_w2 = D_w2
        self.D_b1 = D_b1
        self.D_b2 = D_b2

    def pred_score(self, pred_data):
        self.score = torch.mm(torch.tanh(torch.mm(pred_data.float(), self.D_w1.float()) + self.D_b1), self.D_w2) + self.D_b2
        return self.score

    def forward(self, pred_data, pred_data_label):
        # pdb.set_trace()
        loss = torch.mean(F.binary_cross_entropy_with_logits(self.pred_score(pred_data), pred_data_label.view(-1, 1).float())) 
        return loss

    def get_reward(self, pred_data):
        reward = (torch.sigmoid(self.pred_score(pred_data)) - 0.5) * 2
        return reward

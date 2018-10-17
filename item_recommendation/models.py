#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
# import torchwordemb
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


class Generator(nn.Module):
    """docstring for GEN"""
    def __init__(self, G_user_embeddings, G_item_embeddings, G_item_bias):
        super(Generator, self).__init__()
        self.G_user_embeddings = G_user_embeddings
        self.G_item_embeddings= G_item_embeddings
        self.G_item_bias = G_item_bias

    def all_rating(self, user_index):
        u_embedding = self.G_user_embeddings[user_index, :]
        item_embeddings = self.G_item_embeddings

        all_rating = torch.mm(u_embedding.view(-1, 5), item_embeddings.t()) + self.G_item_bias
        return all_rating

    def all_logits(self, user_index):
        u_embedding = self.G_user_embeddings[user_index]
        item_embeddings = self.G_item_embeddings
        G_item_bias = self.G_item_bias

        score = torch.sum(u_embedding*item_embeddings, 1) + G_item_bias
        return score

    def forward(self, user_index, sample, reward):
        u_embedding = self.G_user_embeddings[user_index]
        item_embeddings = self.G_item_embeddings[sample, :]
        G_item_bias = self.G_item_bias[sample]

        softmax_score = F.softmax(self.all_logits(user_index).view(1, -1), -1)
        gan_prob = torch.gather(softmax_score.view(-1), 0, sample.long()).clamp(min=1e-8)
        loss = -torch.mean(torch.log(gan_prob) * reward)

        return loss


class Discriminator(nn.Module):
    def __init__(self, D_user_embeddings, D_item_embeddings, D_item_bias):
        super(Discriminator, self).__init__()
        self.D_user_embeddings = D_user_embeddings
        self.D_item_embeddings = D_item_embeddings
        self.D_item_bias = D_item_bias

    def pre_logits(self, input_user, input_item):
        u_embedding = self.D_user_embeddings[input_user, :]
        item_embeddings = self.D_item_embeddings[input_item, :]
        D_item_bias = self.D_item_bias[input_item]

        score = torch.sum(u_embedding*item_embeddings, 1) + D_item_bias
        return score

    def forward(self, input_user, input_item, pred_data_label):
        loss = F.binary_cross_entropy_with_logits(self.pre_logits(input_user, input_item), pred_data_label.float())
        return loss

    def get_reward(self, user_index, sample):
        u_embedding = self.D_user_embeddings[user_index, :]
        item_embeddings = self.D_item_embeddings[sample, :]
        D_item_bias = self.D_item_bias[sample]

        reward_logits = torch.sum(u_embedding*item_embeddings, 1) + D_item_bias
        reward = 2 * (torch.sigmoid(reward_logits) - 0.5)
        return reward


    


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
        # loss = -torch.mean(torch.log(gan_prob) * reward.view(-1) * important_sampling.view(-1))
        # if torch.isnan(loss):
        #     pdb.set_trace()
        loss = -torch.mean(torch.log(gan_prob) * reward * important_sampling)
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


# users = [1, 1, 1, 1] # 表示用户1
# items = [34, 33, 75, 53] #表示各种物品
# labels = [1, 0, 0, 1] # 表示该用户对这个物品的喜好程度。
# socre = discriminator(users, items)
# binary_cross_entropy_with_logits(socre, labels)

# Discriminator
# 先根据 IRGAN 原始论文，依旧使用 矩阵分解格式
class Discriminator(nn.Module):
    def __init__(self, itemNum, userNum, emb_dim, alpha=0.1):
        super(Discriminator, self).__init__()
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.alpha = alpha
        # embedding 
        self.u_embeds = nn.Embedding(userNum, emb_dim)
        self.i_embeds = nn.Embedding(itemNum, emb_dim)
        self.i_bias = Parameter(torch.Tensor(itemNum))
 
        self.reset_parameters()
        
    def forward(self, user, item, label):
        item_v = self.i_embeds(item) # b x e
        user_v = self.u_embeds(user) # b x e
        ibias = self.i_bias[item]
        # IRGAN中用的是 elements wise product, 我这里尝试使用 inner product
        pre_logits = (torch.sum(torch.mm(user_v, item_v.t()), 1) + ibias) # bxe * exb -> b
        # 直接在模型中把 loss 计算得了...
        loss = F.binary_cross_entropy_with_logits(pre_logits, label) + self.alpha * (torch.norm(item_v) + torch.norm(user_v) + torch.norm(ibias))
        return loss 
    
    # 计算 reward
    def getReward(self, user, item):
        # 获得reward的过程。
        base_line = 0.5
        reward_logits = torch.mm(self.u_embeds(user), self.i_embeds(item).t()) + self.i_bias[item] # 1xe * exb -> 1xb
        
        reward = 2 * (torch.sigmoid(reward_logits) - base_line)
        reward = reward.view(-1, 1)
        return reward
    
    def reset_parameters(self):
        self.i_bias.data.uniform_(-1, 1)

 # Generator
class Generator(nn.Module):
    def __init__(self, itemNum, userNum, emb_dim, user_pos_train):
        super(Generator, self).__init__()
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        
        # user-item 对
        self.user_pos_train = user_pos_train
        
        # embedding 
        self.u_embeds = nn.Embedding(userNum, emb_dim)
        self.i_embeds = nn.Embedding(itemNum, emb_dim)
        self.i_bias = Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
        
    def forward(self, user, dis):
        sample_lambda = 0.2
        
        # 获取 postive 样本的数量
        pos = user_pos_train[user]
        
        # 获取该user 对所有 item 的 rating
        rating = torch.mm(self.u_embeds(user), self.i_embeds.weight.t()) + self.i_bias # 1xe * exitem -> 1xitem
        exp_rating = torch.exp(rating)
        prob = exp_rating / torch.sum(exp_rating) # 我们把这个看作是 generator 的 distribution
        prob = prob.view(-1, 1)
        
        # 接下来是重要性采样
        pn = (1 - sample_lambda) * prob
        pn[pos] += sample_lambda * 1.0 / len(pos)
        sample = np.random.choice(np.arange(self.itemNum), 2 * len(pos), p=pn.data.numpy().reshape(-1))
        
        # 调用 discrimitor 的 getReward 获取 reward
        reward = dis.getReward(user, Variable(torch.from_numpy(sample).type(torch.LongTensor)))
        reward = reward * prob[torch.from_numpy(sample).type(torch.LongTensor)] / pn[torch.from_numpy(sample).type(torch.LongTensor)]
        
        # 跟discriminator一样，我们在这里面根据 loss 来 update generator
        
        # loss of generator
        i_prob = prob[torch.from_numpy(sample).type(torch.LongTensor)]
        gen_loss = -torch.mean(torch.log(i_prob) * reward)
        
        return gen_loss
        
    def reset_parameters(self):
        self.i_bias.data.uniform_(-1, 1)
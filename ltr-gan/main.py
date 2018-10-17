import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import pdb
import torch.nn.functional as F
from models import *
import utils as ut
from eval.precision import precision_at_k
from eval.ndcg import ndcg_at_k
from eval.map import MAP
from eval.mrr import MRR

# =============================================================================
device = [6]
FEATURE_SIZE = 46
HIDDEN_SIZE = 46
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
D_LEARNING_RATE = 0.001
G_LEARNING_RATE = 0.001
TEMPERATURE = 0.2
LAMBDA = 0.5
os.makedirs('saved_models', exist_ok=True)
# torch.cuda.synchronize()
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# =============================================================================
with torch.cuda.device(device[0]):
    # models

    G_w1 = torch.autograd.Variable((torch.ones(FEATURE_SIZE, HIDDEN_SIZE)).cuda(), requires_grad=True)
    G_w2 = torch.autograd.Variable((torch.ones(HIDDEN_SIZE, 1)).cuda(), requires_grad=True)
    G_b1 = torch.autograd.Variable(torch.zeros(HIDDEN_SIZE).cuda(), requires_grad=True)
    G_b2 = torch.autograd.Variable(torch.zeros(1).cuda(), requires_grad=True)
    
    # the hyperparameter in D
    D_w1 = torch.autograd.Variable((torch.ones(FEATURE_SIZE, HIDDEN_SIZE)).cuda(), requires_grad=True)
    D_w2 = torch.autograd.Variable((torch.ones(HIDDEN_SIZE, 1)).cuda(), requires_grad=True)
    D_b1 = torch.autograd.Variable(torch.zeros(HIDDEN_SIZE).cuda(), requires_grad=True)
    D_b2 = torch.autograd.Variable(torch.zeros(1).cuda(), requires_grad=True)

    criterion = torch.nn.DataParallel(L2Loss(), device_ids=device)
    criterion.cuda()

DG_init = [G_w1, G_w2, D_w1, D_w2]
for param in DG_init:
    torch.nn.init.normal_(param, mean=0, std=0.1)

generator = torch.nn.DataParallel(GEN(G_w1, G_w2, G_b1, G_b2, TEMPERATURE), device_ids=device)
generator.cuda()
discriminator = torch.nn.DataParallel(DIS(D_w1, D_w2, D_b1, D_b2), device_ids=device)
discriminator.cuda()

DG_param = [G_w1, G_w2, G_b1, G_b2, D_w1, D_w2, D_b1, D_b2]

G_params = list(generator.parameters()) + [G_w1, G_w2, G_b1, G_b2]
optimizer_G = torch.optim.SGD(G_params, lr = G_LEARNING_RATE, momentum=0.9)
D_params = list(discriminator.parameters()) + [D_w1, D_w2, D_b1, D_b2]
optimizer_D = torch.optim.SGD(D_params, lr = D_LEARNING_RATE, momentum=0.9)

workdir = 'MQ2008-semi'
DIS_TRAIN_FILE = workdir + '/run-train-gan-1.txt'
GAN_MODEL_BEST_FILE = workdir + '/gan_best_nn.model'


query_url_feature, query_url_index, query_index_url =\
    ut.load_all_query_url_feature(workdir + '/Large_norm.txt', FEATURE_SIZE)
query_pos_train = ut.get_query_pos(workdir + '/train.txt')
query_pos_test = ut.get_query_pos(workdir + '/test.txt')

def generate_for_d(filename):
    data = []
    print('negative sampling for d using g ...')
    for query in query_pos_train:
        pos_list = query_pos_train[query]
        all_list = query_index_url[query]
        candidate_list = all_list

        candidate_list_feature = [query_url_feature[query][url] for url in candidate_list]
        candidate_list_feature = np.asarray(candidate_list_feature)
        with torch.cuda.device(device[0]):
            candidate_list_score = generator.module.pred_score(torch.tensor(candidate_list_feature).cuda())
        # softmax for candidate
        candidate_list_score = candidate_list_score.detach().cpu().numpy()
        exp_rating = np.exp(candidate_list_score - np.max(candidate_list_score))
        prob = exp_rating / np.sum(exp_rating)

        neg_list = np.random.choice(candidate_list, size=[len(pos_list)], p=prob.reshape(-1,))

        for i in range(len(pos_list)):
            data.append((query, pos_list[i], neg_list[i]))

    random.shuffle(data)
    with open(filename, 'w') as fout:
        for (q, pos, neg) in data:
            fout.write(','.join([str(f) for f in query_url_feature[q][pos]])
                       + '\t'
                       + ','.join([str(f) for f in query_url_feature[q][neg]]) + '\n')
            fout.flush()


def main():
    p_best_val = 0.0
    ndcg_best_val = 0.0

    for epoch in range(30):
        if epoch >= 0:
            print('Training D ...')
            for d_epoch in range(100):
                if d_epoch % 30 == 0:
                        generate_for_d(DIS_TRAIN_FILE)
                        train_size = ut.file_len(DIS_TRAIN_FILE)
                
                index = 1
                while True:
                    if index > train_size:
                        break
                    if index + BATCH_SIZE <= train_size + 1:
                        input_pos, input_neg = ut.get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
                    else:
                        input_pos, input_neg = ut.get_batch_data(DIS_TRAIN_FILE, index, train_size - index + 1)
                    index += BATCH_SIZE

                    pred_data = []
                    pred_data.extend(input_pos)
                    pred_data.extend(input_neg)
                    pred_data = np.asarray(pred_data)

                    pred_data_label = [1.0] * len(input_pos)
                    pred_data_label.extend([0.0] * len(input_neg))
                    pred_data_label = np.asarray(pred_data_label)

                    loss_d = discriminator(torch.tensor(pred_data), torch.tensor(pred_data_label)) \
                            + WEIGHT_DECAY * (criterion(D_w1) + criterion(D_w2)
                                           + criterion(D_b1) + criterion(D_b2))
                    optimizer_D.zero_grad()
                    loss_d.backward()
                    optimizer_D.step()
                print("\r[D Epoch %d/%d] [loss: %f]" %(d_epoch, 100, loss_d.item()))

        print('Training G ...')
        for g_epoch in range(30):
            num = 0
            for query in query_pos_train.keys():
                pos_list = query_pos_train[query]
                pos_set = set(pos_list)
                all_list = query_index_url[query]

                all_list_feature = [query_url_feature[query][url] for url in all_list]
                all_list_feature = np.asarray(all_list_feature)
                # pdb.set_trace()
                with torch.cuda.device(device[0]):
                    all_list_score = generator.module.pred_score(torch.tensor(all_list_feature).cuda())
                all_list_score = all_list_score.detach().cpu().numpy()
                # softmax for all
                exp_rating = np.exp(all_list_score - np.max(all_list_score))
                prob = exp_rating / np.sum(exp_rating)
                
                prob_IS = prob * (1.0 - LAMBDA)

                for i in range(len(all_list)):
                    if all_list[i] in pos_set:
                        prob_IS[i] += (LAMBDA / (1.0 * len(pos_list)))
                # pdb.set_trace()
                choose_index = np.random.choice(np.arange(len(all_list)), [5 * len(pos_list)], p=prob_IS.reshape(-1,))
                choose_list = np.array(all_list)[choose_index]
                choose_feature = [query_url_feature[query][url] for url in choose_list]
                choose_IS = np.array(prob)[choose_index] / np.array(prob_IS)[choose_index]
                
                choose_index = np.asarray(choose_index)
                choose_feature = np.asarray(choose_feature)
                choose_IS = np.asarray(choose_IS)
                with torch.cuda.device(device[0]):
                    choose_reward = discriminator.module.get_reward(torch.tensor(choose_feature).cuda())
                choose_reward.detach_()

                loss_g = generator(torch.tensor(all_list_feature).cuda(), torch.tensor(choose_index), choose_reward, torch.tensor(choose_IS)) \
                        + WEIGHT_DECAY * (criterion(G_w1) + criterion(G_w2)
                                   + criterion(G_b1) + criterion(G_b2))
                # pdb.set_trace()

                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()
                num += 1
                # if num == 200:
                #     pdb.set_trace()
            print("\r[G Epoch %d/%d] [loss: %f]" %(g_epoch, 30, loss_g.item()))
            # pdb.set_trace()
            p_5 = precision_at_k(device, generator, query_pos_test, query_pos_train, query_url_feature, k=5)
            ndcg_5 = ndcg_at_k(device, generator, query_pos_test, query_pos_train, query_url_feature, k=5)

            if p_5 > p_best_val:
                p_best_val = p_5
                ndcg_best_val = ndcg_5
                print("Best:", "gen p@5 ", p_5, "gen ndcg@5 ", ndcg_5)
            elif p_5 == p_best_val:
                if ndcg_5 > ndcg_best_val:
                    ndcg_best_val = ndcg_5
                    print("Best:", "gen p@5 ", p_5, "gen ndcg@5 ", ndcg_5)
            #validation
            # p_5 = precision_at_k(val_loader, 5)
            # if p_5 > p_best_val:
            #     p_best_val = p_5
            #     print("Best:", "gen p@5 ", p_5)
            #     torch.save(recipe_emb.state_dict(), 'saved_models/recipe_emb_%d_%.3f.pth' % (epoch, p_5))
            #     param_num = 1
            #     for param in DG_param:
            #         torch.save(param, 'saved_models/param%d_%d_%.3f.pt' % (param_num, epoch, p_5))
            #         param_num += 1
    p_1_best = precision_at_k(device, generator, query_pos_test, query_pos_train, query_url_feature, k=1)
    p_3_best = precision_at_k(device, generator, query_pos_test, query_pos_train, query_url_feature, k=3)
    p_5_best = precision_at_k(device, generator, query_pos_test, query_pos_train, query_url_feature, k=5)
    p_10_best = precision_at_k(device, generator, query_pos_test, query_pos_train, query_url_feature, k=10)

    ndcg_1_best = ndcg_at_k(device, generator, query_pos_test, query_pos_train, query_url_feature, k=1)
    ndcg_3_best = ndcg_at_k(device, generator, query_pos_test, query_pos_train, query_url_feature, k=3)
    ndcg_5_best = ndcg_at_k(device, generator, query_pos_test, query_pos_train, query_url_feature, k=5)
    ndcg_10_best = ndcg_at_k(device, generator, query_pos_test, query_pos_train, query_url_feature, k=10)

    # map_best = MAP(sess, generator, query_pos_test, query_pos_train, query_url_feature)
    # mrr_best = MRR(sess, generator, query_pos_test, query_pos_train, query_url_feature)

    print("Best ", "p@1 ", p_1_best, "p@3 ", p_3_best, "p@5 ", p_5_best, "p@10 ", p_10_best)
    print("Best ", "ndcg@1 ", ndcg_1_best, "ndcg@3 ", ndcg_3_best, "ndcg@5 ", ndcg_5_best, "p@10 ", ndcg_10_best)
    # print("Best MAP ", map_best)
    # print("Best MRR ", mrr_best)
if __name__ == '__main__':
    main()
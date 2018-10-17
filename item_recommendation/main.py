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
import pickle
import torch.nn.functional as F
from models import *
import utils as ut
import multiprocessing


cores = multiprocessing.cpu_count()

#########################################################################################
# Hyper-parameters
#########################################################################################
EMB_DIM = 5
USER_NUM = 943
ITEM_NUM = 1683
BATCH_SIZE = 16
INIT_DELTA = 0.05

all_items = set(range(ITEM_NUM))
workdir = 'ml-100k/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = [0]

#########################################################################################
# Load data
#########################################################################################
user_pos_train = {}
with open(workdir + 'movielens-100k-train.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]

user_pos_test = {}
with open(workdir + 'movielens-100k-test.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]

# all_users = user_pos_train.keys()
# all_users.sort()

print("load model...")
with open("ml-100k/model_dns_ori.pkl", "rb") as f:
    param = pickle.load(f, encoding='latin1')

with torch.cuda.device(device[0]):

    G_user_embeddings = torch.autograd.Variable(torch.tensor(param[0]).cuda(), requires_grad=True)
    G_item_embeddings = torch.autograd.Variable(torch.tensor(param[1]).cuda(), requires_grad=True)
    G_item_bias = torch.autograd.Variable(torch.tensor(param[2]).cuda(), requires_grad=True)
    
    D_user_embeddings = torch.autograd.Variable((torch.ones(USER_NUM, EMB_DIM)).cuda(), requires_grad=True)
    D_item_embeddings = torch.autograd.Variable((torch.ones(ITEM_NUM, EMB_DIM)).cuda(), requires_grad=True)
    D_item_bias = torch.autograd.Variable((torch.zeros(ITEM_NUM)).cuda(), requires_grad=True)

    criterion = torch.nn.DataParallel(L2Loss(), device_ids=device)
    criterion.cuda()

DG_init = [D_user_embeddings, D_item_embeddings]
for params in DG_init:
    torch.nn.init.uniform_(params, a=0.05, b=-0.05)

generator = torch.nn.DataParallel(Generator(G_user_embeddings, G_item_embeddings, G_item_bias), device_ids=device)
generator.cuda()
discriminator = torch.nn.DataParallel(Discriminator(D_user_embeddings, D_item_embeddings, D_item_bias), device_ids=device)
discriminator.cuda()

G_params = list(generator.parameters()) + [G_user_embeddings, G_item_embeddings, G_item_bias]
optimizer_G = torch.optim.SGD(G_params, lr = 0.001, momentum=0.9)
D_params = list(discriminator.parameters()) + [D_user_embeddings, D_item_embeddings, D_item_bias]
optimizer_D = torch.optim.SGD(D_params, lr = 0.001, momentum=0.9)
lamda=0.1 / BATCH_SIZE

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def simple_test_one_user(x):
    # import pdb; pdb.set_trace()
    rating = x[0]
    u = x[1]

    test_items = list(all_items - set(user_pos_train[u]))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])

def simple_test(model):
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    test_users = list(user_pos_test.keys())
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        user_batch_rating = generator.module.all_rating(user_batch)
        user_batch_rating = user_batch_rating.detach_().cpu().numpy()

        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret

def generate_for_d(model, filename):
    data = []
    
    for u in user_pos_train:
        pos = user_pos_train[u]
        
        rating = generator.module.all_rating(u)
        rating = rating.detach_().cpu().numpy()

        rating = np.array(rating) / 0.2  # Temperature
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)

        neg = np.random.choice(np.arange(ITEM_NUM), size=len(pos), p=prob.reshape(-1,))
        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))

dis_log = open(workdir + 'dis_log.txt', 'w')
gen_log = open(workdir + 'gen_log.txt', 'w')

def main():

    best = 0.
    gen_log = open(workdir + 'gen_log.txt', 'w')
    for epoch in range(15):
        if epoch >= 0:
            for d_epoch in range(100):
                if d_epoch % 5 == 0:
                    generate_for_d(generator, DIS_TRAIN_FILE)
                    train_size = ut.file_len(DIS_TRAIN_FILE)
                index = 1
                while True:
                    if index > train_size:
                        break
                    if index + BATCH_SIZE <= train_size + 1:
                        input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index, BATCH_SIZE)
                    else:
                        input_user, input_item, input_label = ut.get_batch_data(DIS_TRAIN_FILE, index,
                                                                                train_size - index + 1)
                    index += BATCH_SIZE
                    # pre_logits = discriminator.module.pre_logits(input_user, input_item)
                    D_loss = discriminator(input_user, input_item, torch.tensor(input_label)) \
                            + lamda * (criterion(D_user_embeddings) + criterion(D_item_embeddings) + criterion(D_item_bias))

                    optimizer_D.zero_grad()
                    D_loss.backward()
                    optimizer_D.step()
                print("\r[D Epoch %d/%d] [loss: %f]" %(d_epoch, 100, D_loss.item()))

            for g_epoch in range(50):
                for u in user_pos_train:
                    sample_lambda = 0.2
                    pos = user_pos_train[u]
                    rating = generator.module.all_logits(u)
                    rating = rating.detach_().cpu().numpy()

                    exp_rating = np.exp(rating)
                    prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

                    pn = (1 - sample_lambda) * prob
                    pn[pos] += sample_lambda * 1.0 / len(pos)
                    # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta

                    sample = np.random.choice(np.arange(ITEM_NUM), 2 * len(pos), p=pn)
                    ###########################################################################
                    # Get reward and adapt it with importance sampling
                    ###########################################################################
                    reward = discriminator.module.get_reward(u, sample)
                    reward = reward.detach_().cpu().numpy() * prob[sample] / pn[sample]
                    ###########################################################################
                    # Update G
                    ###########################################################################
                    with torch.cuda.device(device[0]):
                        G_loss = generator(u, torch.tensor(sample), torch.tensor(reward))
                    optimizer_G.zero_grad()
                    G_loss.backward()
                    optimizer_G.step()
                print("\r[G Epoch %d/%d] [loss: %f]" %(g_epoch, 50, G_loss.item()))
                result = simple_test(generator)
                print("epoch ", epoch, "gen: ", result)
                buf = '\t'.join([str(x) for x in result])
                gen_log.write(str(epoch) + '\t' + buf + '\n')
                gen_log.flush()

    gen_log.close()

if __name__ == '__main__':
    main()
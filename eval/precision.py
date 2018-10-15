import numpy as np
import torch


def precision_at_k(device, generator, query_pos_test, query_pos_train, query_url_feature, k=5):
    p = 0.0
    cnt = 0
    for query in query_pos_test.keys():
        pos_set = set(query_pos_test[query])
        pred_list = list(set(query_url_feature[query].keys()) - set(query_pos_train.get(query, [])))
        if len(pred_list) < k:
            continue

        pred_list_feature = [query_url_feature[query][url] for url in pred_list]
        pred_list_feature = np.asarray(pred_list_feature)
        with torch.cuda.device(device[0]):
            pred_list_score = generator.module.pred_score(torch.tensor(pred_list_feature).cuda())
        pred_list_score = pred_list_score.detach().cpu().numpy()
        pred_url_score = zip(pred_list, pred_list_score)
        pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)

        num = 0.0
        for i in range(0, k):
            (url, score) = pred_url_score[i]
            if url in pos_set:
                num += 1.0
        num /= (k * 1.0)

        p += num
        cnt += 1

    return p / float(cnt)


def precision_at_k_user(device, generator, query_pos_test, query_pos_train, query_url_feature, k=5):
    p_list = []
    query_test_list = sorted(query_pos_test.keys())
    for query in query_test_list:
        pos_set = set(query_pos_test[query])
        pred_list = list(set(query_url_feature[query].keys()) - set(query_pos_train.get(query, [])))
        if len(pred_list) < k:
            continue

        pred_list_feature = [query_url_feature[query][url] for url in pred_list]
        pred_list_feature = np.asarray(pred_list_feature)
        with torch.cuda.device(device[0]):
            pred_list_score = generator.module.pred_score(torch.tensor(pred_list_feature).cuda())
        pred_list_score = pred_list_score.detach().cpu().numpy()
        pred_url_score = zip(pred_list, pred_list_score)
        pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)

        num = 0.0
        for i in range(0, k):
            (url, score) = pred_url_score[i]
            if url in pos_set:
                num += 1.0
        num /= (k * 1.0)

        p_list.append(num)

    return p_list
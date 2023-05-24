import time, copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from src.datasets import *
from src.helper import *

def train(kwargs, max_lst, model_name, embed_model, model, loss_fn1, loss_fn2, loss_fn3, loss_fn4, embed_optimizer, model_optimizer, fgm, data, dev_data, threshold):
    for epoch in range(kwargs["epochs"]):
        embed_model.train()
        model.train()
        start_time = time.time()
        print("Model is training in epoch {}".format(epoch))
        loss_all = 0.
        level1_pred, level2_pred, level3_pred, levels_pred, levels_pred_ = [], [], [], [], []
        level1_labels, level2_labels, level3_labels, levels_labels = [], [], [], []
        for batch in tqdm(data, desc='Training', colour = 'MAGENTA'):
            embed_model.zero_grad()
            model.zero_grad()
            args = to_tensor(batch)
            att_input, pooled_emb = embed_model(**args)

            local_logits, global_logits = model(att_input, pooled_emb)
            local_logits[0] = local_logits[0].cpu()
            local_logits[1] = local_logits[1].cpu()
            local_logits[2] = local_logits[2].cpu()
            global_logits = global_logits.cpu()
            level1, level2, level3, levels  = args['level1'], args['level2'], args['level3'], args['levels']
            loss_1, loss_2, loss_3 = loss_fn1(local_logits[0], level1.float()), loss_fn2(local_logits[1], level2.float()), loss_fn3(local_logits[2], level3.float())#, loss_fn4(global_logits, levels.float())
            if epoch < 5:
                loss = loss_1 + loss_2 + loss_3
            else:
                loss = 0.05*loss_1 + loss_2 + 4*loss_3

            pred_1, pred_2, pred_3, pred_s, pred_s_ = get_preds(kwargs, local_logits[0], local_logits[1], local_logits[2], global_logits)  # numpy
            level1_pred.extend(pred_1)
            level2_pred.extend(pred_2)
            level3_pred.extend(pred_3)
            levels_pred.extend(pred_s)
            levels_pred_.extend(pred_s_)  # 2022.5.4 global scores 
            level1_labels.extend(level1.detach().numpy())
            level2_labels.extend(level2.detach().numpy())
            level3_labels.extend(level3.detach().numpy())
            levels_labels.extend(levels.detach().numpy())  # 2022.5.4 global scores 
            loss_all += loss.item()

            embed_optimizer.zero_grad()
            model_optimizer.zero_grad()
            loss.backward()

            # 2022.5.18 对抗训练
            fgm.attack() # 在embedding上添加对抗扰动
            att_input, pooled_emb = embed_model(**args)
            local_logits, global_logits = model(att_input, pooled_emb)
            local_logits[0] = local_logits[0].cpu()
            local_logits[1] = local_logits[1].cpu()
            local_logits[2] = local_logits[2].cpu()
            loss_1, loss_2, loss_3 = loss_fn1(local_logits[0], level1.float()), loss_fn2(local_logits[1], level2.float()), loss_fn3(local_logits[2], level3.float())
            if epoch < 5:
                loss_adv = loss_1 + loss_2 + loss_3
            else:
                loss_adv = 0.05*loss_1 + loss_2 + 4*loss_3
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore() # 恢复embedding参数

            embed_optimizer.step()
            model_optimizer.step()

        end_time = time.time()
        print(" took: {:.1f} min".format((end_time - start_time)/60.))
        print("TRAINED for {} epochs".format(epoch))

        all_preds = [level1_pred, level2_pred, level3_pred, levels_pred, levels_pred_]
        all_labels = [level1_labels, level2_labels, level3_labels, levels_labels]
        if epoch >= kwargs["num_warm"]:
            # print("training loss: loss={}".format(loss_all/len(data)))
            trn_scores = get_scores(all_preds, all_labels, loss_all, len(data), data_name="TRAIN")
            dev_scores, _ = eval(kwargs, embed_model, model, loss_fn1, loss_fn2, loss_fn3, loss_fn4, dev_data, threshold, data_name='DEV')
            f = open('{}all/{}.all_scores.txt'.format(kwargs["result_path"], model_name), 'a')
            f.write(' ==================================================  Epoch: {}  ==================================================\n'.format(epoch))
            f.write('TrainScore: \n{}\nEvalScore: \n{}\n'.format(json.dumps(trn_scores), json.dumps(dev_scores))) 
            save_best(kwargs["result_path"], kwargs["checkpoint_path"], model_name, embed_model, model, dev_scores, kwargs["score_key"], epoch, max_lst)
        print("ALLTRAINED for {} epochs".format(epoch))

def eval(kwargs, embed_model, model, loss_fn1, loss_fn2, loss_fn3, loss_fn4, data, threshold, data_name="VAL"):
    level1_pred, level2_pred, level3_pred, levels_pred, levels_pred_ = [], [], [], [], []
    level1_labels, level2_labels, level3_labels, levels_labels = [], [], [], []
    embed_model.eval()
    model.eval()
    loss_all = 0.
    for batch in tqdm(data, desc='Evaling', colour = 'CYAN'):
        with torch.no_grad():
            args = to_tensor(batch)
            att_input, pooled_emb = embed_model(**args)
            local_logits, global_logits = model(att_input, pooled_emb)
            local_logits[0] = local_logits[0].cpu()
            local_logits[1] = local_logits[1].cpu()
            local_logits[2] = local_logits[2].cpu()
            global_logits = global_logits.cpu()
            level1, level2, level3, levels  = args['level1'], args['level2'], args['level3'], args['levels']
            loss_1, loss_2, loss_3 = loss_fn1(local_logits[0], level1.float()), loss_fn2(local_logits[1], level2.float()), loss_fn3(local_logits[2], level3.float())#, loss_fn4(global_logits, levels.float())
            loss = loss_1 + loss_2 + loss_3

            pred_1, pred_2, pred_3, pred_s, pred_s_ = get_preds(kwargs, local_logits[0], local_logits[1], local_logits[2], global_logits)  # numpy
            level1_pred.extend(pred_1)
            level2_pred.extend(pred_2)
            level3_pred.extend(pred_3)
            levels_pred.extend(pred_s)
            levels_pred_.extend(pred_s_)  # 2022.5.4 global scores 
            level1_labels.extend(level1.detach().numpy())
            level2_labels.extend(level2.detach().numpy())
            level3_labels.extend(level3.detach().numpy())
            levels_labels.extend(levels.detach().numpy())  # 2022.5.4 global scores 
            loss_all += loss.item()
    all_preds = [level1_pred, level2_pred, level3_pred, levels_pred, levels_pred_]
    all_labels = [level1_labels, level2_labels, level3_labels, levels_labels]
    dev_scores = get_scores(all_preds, all_labels, loss_all, len(data), data_name=data_name)

    # 2022.5.14 测试验证集（topk， threshold）
    # result_path = 'results/'
    # f = open('{}all/test_topk_threshold_scores.txt'.format(result_path), 'a')
    # f.write('topk: {} {} {}  threshold: {} {} {}  scores: {}\n'.format(kwargs["top_k"][0], kwargs["top_k"][1], kwargs["top_k"][2], kwargs["threshold"][0], kwargs["threshold"][1], kwargs["threshold"][2], dev_scores['levels_F1_macro']))
    # f.write('EvalScore: \n{}\n'.format(json.dumps(dev_scores))) 

    return dev_scores, levels_pred

# 2022-5-1 top-k
def get_preds(kwargs, logit1, logit2, logit3, logits):
    threshold_1 = kwargs["threshold"][0]
    threshold_2 = kwargs["threshold"][1]
    threshold_3 = kwargs["threshold"][2]
    pre_pred_1 = torch.sigmoid(logit1)
    top_1 = get_label_topk(pre_pred_1, kwargs["top_k"][0])
    pred_1 = get_threshold(pre_pred_1, threshold_1, top_1)
    # pred_1_ = get_threshold(pre_pred_1, threshold_1-0.5, top_1)

    if kwargs["if_label_mask"]:
        mask_2 = get_mask(kwargs["key1_2_path"], pred_1, kwargs["class_num"][1])
        pred_2 = mask_2 * torch.sigmoid(logit2)
        top_2 = get_label_topk(pred_2, kwargs["top_k"][1])
        pred_2 = get_threshold(pred_2, threshold_2, top_2)
        mask_3 = get_mask(kwargs["key2_3_path"], pred_2, kwargs["class_num"][2])
        pred_3 = mask_3 * torch.sigmoid(logit3)
        top_3 = get_label_topk(pred_3, kwargs["top_k"][2])
        pred_3 = get_threshold(pred_3, threshold_3, top_3)        
        pred_1 = pred_1.detach().numpy()
        pred_2 = pred_2.detach().numpy()
        pred_3 = pred_3.detach().numpy()
    else:
        pred_1 = pred_1.detach().numpy()
        pred_2 = get_threshold(torch.sigmoid(logit2), threshold_2).detach().numpy()
        pred_3 = get_threshold(torch.sigmoid(logit3), threshold_3).detach().numpy()

    # 选择 global_logits 还是 local_logits作为最终输出
    pred_s = np.concatenate((pred_1, pred_2, pred_3),axis=1)
    pred_s_ = get_threshold(torch.sigmoid(logits), threshold_1).detach().numpy()  # 2022.5.4 global scores 
    return pred_1, pred_2, pred_3, pred_s, pred_s_

# 2022.4.23 calculate scores
def get_scores(all_preds, all_lebels, loss_all, len, data_name):
    score_dict = dict()
    level1_f1 = f1_score(all_preds[0], all_lebels[0], average='micro')
    level2_f1 = f1_score(all_preds[1], all_lebels[1], average='micro')
    level3_f1 = f1_score(all_preds[2], all_lebels[2], average='micro')
    levels_f1 = f1_score(all_preds[3], all_lebels[3], average='micro')
    levels_pre = precision_score(all_preds[3], all_lebels[3], average='micro')
    levels_recall = recall_score(all_preds[3], all_lebels[3], average='micro')

    score_dict['level1_F1_macro'] = level1_f1
    score_dict['level2_F1_macro'] = level2_f1
    score_dict['level3_F1_macro'] = level3_f1
    score_dict['levels_F1_macro'] = levels_f1
    score_dict['levels_precision'] = levels_pre
    score_dict['levels_recall'] = levels_recall
    score_dict['all_loss'] = loss_all/len
    print("Evaling on \"{}\" data".format(data_name))
    for s_name, s_val in score_dict.items():  
        print("{}: {}".format(s_name, s_val)) 
    return score_dict

def save_best(result_path, checkpoint_path, model_name, embed_model, model, scores, score_key, epoch, max_lst):
    '''
    Save the top-5 model scores and the best model parameters
    '''
    scores = copy.deepcopy(scores)  
    curr_score = scores[score_key]
    score_updated = False
    if len(max_lst) < 5:
        score_updated = True
        if len(max_lst) > 0:
            prev_max = max_lst[-1][0][score_key]  # prev_max is current max score
        else:
            prev_max = curr_score
        max_lst.append((scores, epoch)) 
    else:
        if curr_score > max_lst[0][0][score_key]: 
            score_updated = True
            prev_max = max_lst[-1][0][score_key]  
            max_lst[0] = (scores, epoch)  #  replace smallest score
        else:
            prev_max = max_lst[-1][0][score_key] 
    print('The epoch_{} {}: {}\nCurrent max {}: {}'.format(epoch, score_key, curr_score, score_key, prev_max))

    # update best saved model and file with top scores
    if score_updated:
        max_lst = sorted(max_lst, key=lambda p: p[0][score_key])  # lowest first 
        # write top 5 scores
        f = open('{}top5/{}.top5_{}.txt'.format(result_path, model_name, score_key), 'w')  # overrides
        for p in max_lst:
            f.write('Epoch: {}\nScore: {}\nAll Scores: {}\n'.format(p[1], p[0][score_key],
                                                                        json.dumps(p[0])))
        # save best model step, if its this one
        if curr_score > prev_max or epoch == 0:
            torch.save({
            'epoch': epoch,
            'embed_model_state_dict': embed_model.state_dict(),
            'model_state_dict': model.state_dict(),
            }, '{}ckp-{}-{}.tar'.format(checkpoint_path, model_name, 'BEST'))
from importlib import import_module
from src.datasets import *
import os
from os import path
import numpy as np
from train_eval import train, eval
import argparse
from src.Models import *

from ray import tune
from ray.tune.schedulers import ASHAScheduler

parser = argparse.ArgumentParser(description='Chinese Toxic Classification')
parser.add_argument('--mode', default="train", type=str, help='train/test')
# 是否微调超参数
parser.add_argument('--tune_param', default=False, type=bool, help='True for param tuning')
parser.add_argument('--tune_samples', default=1, type=int, help='Number of tuning experiments to run')
parser.add_argument('--tune_asha', default=False, type=bool, help='If use ASHA scheduler for early stopping')
parser.add_argument('--tune_file', default='RoBERTa', type=str, help='Suffix of filename for parameter tuning results')
parser.add_argument('--tune_gpu', default=True, type=bool, help='Use GPU to tune parameters')
args = parser.parse_args()

# TOC
# search_space = {
#     'learning_rate': tune.choice([1e-5, 5e-5]),
#     'num_epochs': tune.choice([5]),
#     'batch_size': tune.choice([32, 64]),
#     'dropout': tune.choice([0.2, 0.5]),
#     'seed': tune.choice([1]),
#     "pad_size" : tune.choice([50, 100])
# }

# THUCnews
search_space = {
    'learning_rate': tune.choice([1e-5]),
    'num_epochs': tune.choice([5]),
    'batch_size': tune.choice([32]),
    # 'dropout': tune.choice([0.2, 0.3, 0.5]),
    'seed': tune.choice([1]),
    "pad_size" : tune.choice([80]),
    "alpha1" : tune.choice([0.5])
}

def convert_label(preds):
    final_pred = []
    for pred in preds:
        if pred == [1, 0]:
            final_pred.append("offensive")
        else:
            final_pred.append("non-offensive")
    return final_pred

if __name__ == '__main__':
    # dataset = 'TOC'  # 数据集
    dataset = "ToxiCN"
    # model_name = "bert-base-chinese"
    model_name = "hfl/chinese-roberta-wwm-ext"
    # model_name = "junnyu/ChineseBERT-base"

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()    
    print("Loading data...")

    x = import_module('model.' + "Config_base")
    config = x.Config_base(model_name, dataset)  # 引入Config参数，包括Config_base和各私有Config

    if not os.path.exists(config.data_path): 
        trn_data = Datasets(config, config.train_path)
        dev_data = Datasets(config, config.dev_path)
        test_data = Datasets(config, config.test_path)
        torch.save({
            'trn_data' : trn_data,
            'dev_data' : dev_data,
            'test_data' : test_data,
            }, config.data_path)
    else:
        checkpoint = torch.load(config.data_path)
        trn_data = checkpoint['trn_data']
        dev_data = checkpoint['dev_data']
        test_data = checkpoint['test_data']
        print('The size of the Training dataset: {}'.format(len(trn_data)))
        print('The size of the Validation dataset: {}'.format(len(dev_data)))
        print('The size of the Test dataset: {}'.format(len(test_data)))
    train_iter = Dataloader(trn_data,  batch_size=int(config.batch_size), SEED=config.seed)
    dev_iter = Dataloader(dev_data,  batch_size=int(config.batch_size), shuffle=False)
    test_iter = Dataloader(test_data,  batch_size=int(config.batch_size), shuffle=False)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    def experiment(tune_config):
        if tune_config:
            for param in tune_config:
                setattr(config, param, tune_config[param])
        train(config, train_iter, dev_iter, test_iter)

    if args.mode == "train":
        if args.tune_param:
            scheduler = ASHAScheduler(metric='metric', mode="max") if args.tune_asha else None        
            analysis = tune.run(experiment, num_samples=args.tune_samples, config=search_space, resources_per_trial={'gpu':int(args.tune_gpu)},
                scheduler=scheduler,
                verbose=3)
            analysis.results_df.to_csv('tune_results_'+args.tune_file+'.csv')
        # if not tune parameters
        else:
            experiment(tune_config=None)

    else:
        embed_model = Bert_Layer(config).to(config.device)
        model = TwoLayerFFNNLayer(config).to(config.device)
        # model_name = "ckp-bert-base-chinese-NN_ML-100_D-0.2_B-64_E-5_Lr-1e-05-BEST.tar"
        model_name = "ckp-bert-base-chinese-NN_ML-150_D-0.5_B-32_E-2_Lr-1e-05-BEST.tar"
        path = '{}/{}'.format(config.checkpoint_path, model_name)
        checkpoint = torch.load(path)
        embed_model.load_state_dict(checkpoint['embed_model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        loss_fn = nn.BCEWithLogitsLoss()
        dev_scores, preds = eval(config, embed_model, model, loss_fn, dev_iter, data_name='TEST')
        print(convert_label(preds))
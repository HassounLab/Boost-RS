import copy
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import itertools
import argparse
import pickle
import numpy as np

np.random.seed(8)
import random

random.seed(8)

import sys
sys.path.append('../')

from sklearn.metrics import roc_auc_score
from util import report_metric
import sklearn.metrics as sk_m
import torch

from tqdm import tqdm


class MLPModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, sigmoid_last_layer=False):
        super(MLPModel, self).__init__()

        # construct layers
        layers = [torch.nn.Linear(input_dim, hidden_dim),
                  torch.nn.ReLU(),
                  torch.nn.Dropout(dropout),
                  torch.nn.Linear(hidden_dim, output_dim)]
        if sigmoid_last_layer:
            layers.append(torch.nn.Sigmoid())

        # construct model
        self.predictor = torch.nn.Sequential(*layers)

    def forward(self, X):
        X = self.predictor(X)
        return X


class RecommenderGNN(torch.nn.Module):
    def __init__(self, num_compound, num_enzyme,
                 hidden_dim, num_layer, dropout=0.5, device='cpu'):
        super(RecommenderGNN, self).__init__()

        # fingerprint and ec dimension
        fp_dim = 167
        ec_dim = 7+68+231

        # embedding layer for compound and enzyme
        if args.aug == 0:
            self.AUG_Embedding = MLPModel(input_dim=fp_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, sigmoid_last_layer=False).to(device)
        elif args.aug == 1:
            self.AUG_Embedding = MLPModel(input_dim=ec_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, sigmoid_last_layer=False).to(device)
        elif args.aug == 2:
            self.AUG_Embedding = MLPModel(input_dim=enzyme_ko_hot.shape[1], hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, sigmoid_last_layer=False).to(device)
        elif args.aug == 3:
            self.AUG_Embedding = MLPModel(input_dim=compound_rclass.shape[1], hidden_dim=hidden_dim, output_dim=hidden_dim,
                                          dropout=dropout, sigmoid_last_layer=False).to(device)
        elif args.aug == 5:
            self.AUG_Embedding = MLPModel(input_dim=fp_dim + ec_dim + enzyme_ko_hot.shape[1] + compound_rclass.shape[1],
                                          hidden_dim=hidden_dim, output_dim=hidden_dim, dropout=dropout, sigmoid_last_layer=False).to(device)
        else:
            raise NotImplementedError

        self.MF_Embedding_Compound = torch.nn.Embedding(num_compound, hidden_dim).to(device)
        self.MF_Embedding_Enzyme = torch.nn.Embedding(num_enzyme, hidden_dim).to(device)
        self.MLP_Embedding_Compound = torch.nn.Embedding(num_compound, hidden_dim).to(device)
        self.MLP_Embedding_Enzyme = torch.nn.Embedding(num_enzyme, hidden_dim).to(device)

        self.dropout = torch.nn.Dropout(p=dropout)

        # main-task: compound-enzyme interaction prediction net. * 2 since concatenation
        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 3, 1),
            torch.nn.Sigmoid()
        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU()
        )

        # save parameters
        self.num_compound = num_compound
        self.num_enzyme = num_enzyme

    def forward(self, compound_ids, enzyme_ids, aug_f):
        aug_embedding = self.AUG_Embedding(aug_f)

        # MF
        mf_embedding_compound = self.MF_Embedding_Compound(compound_ids)
        mf_embedding_enzyme = self.MF_Embedding_Enzyme(enzyme_ids)
        mf_vector = mf_embedding_enzyme * mf_embedding_compound
        # MLP 
        mlp_embedding_compound = self.MLP_Embedding_Compound(compound_ids)
        mlp_embedding_enzyme = self.MLP_Embedding_Enzyme(enzyme_ids)
        mlp_vector = torch.cat([mlp_embedding_enzyme, mlp_embedding_compound], dim=-1)
        mlp_vector = self.fc1(mlp_vector)
        # cat all
        predict_vector = torch.cat([mf_vector, mlp_vector, aug_embedding], dim=-1)

        predict_vector = self.ce_predictor(self.dropout(predict_vector))

        return predict_vector


def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output, 1e-6, 1.0 - 1e-6)

    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def load_data():
    with open('./data/data_collection_2021.pkl', 'rb') as fi:
        data = pickle.load(fi)
    tr_p, va_p, te_p, va_pn, te_pn, n_all_exclusive, num_compound, num_enzyme, compound_i2n, \
    enzyme_i2n, fp_label, ec_label = data['tr_p'], data['va_p'], data['te_p'], data['va_pn'], data['te_pn'], data['n_all_exclusive'],\
                                     data['num_compound'], data['num_enzyme'], data['compound_i2n'], data['enzyme_i2n'], data['fp_label'], data['ec_label']
    return tr_p, va_p, te_p, va_pn, te_pn, n_all_exclusive, num_compound, num_enzyme, compound_i2n, enzyme_i2n, fp_label, ec_label


def read_mt_data(compound_i2n, enzyme_i2n):
    with open('./data/mt_data.pkl', 'rb') as fi:
        data = pickle.load(fi)
    rpairs_pos, cpd_module, cpd_pathway, enzyme_ko, enzyme_ko_hot, enzyme_module, enzyme_pathway = data['rpairs_pos'], data['cpd_module'], data['cpd_pathway'],\
                                          data['enzyme_ko'], data['enzyme_ko_hot'], data['enzyme_module'], data['enzyme_pathway']
    return rpairs_pos, cpd_module, cpd_pathway, enzyme_ko, enzyme_ko_hot, enzyme_module, enzyme_pathway


def train():
    # construct model
    model = RecommenderGNN(num_compound=num_compound, num_enzyme=num_enzyme,
                           hidden_dim=args.hidden_dim, num_layer=2, dropout=args.dropout, device=device).to(device)

    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    val_maps = []
    best_valid_map = 0.0
    best_model_state = None

    # train
    for iter in range(args.epoch):
        model.train()

        tr_p_obj = tr_p
        # sample negative links
        tr_n_ids = np.random.choice(np.arange(n_all_exclusive.shape[0]), tr_p_obj.shape[0] * args.neg_rate)
        tr_n_obj = n_all_exclusive[tr_n_ids]

        tr_obj = torch.cat([tr_p_obj, tr_n_obj], dim=0)
        tr_obj_compound_ids = tr_obj[:, 0]
        tr_obj_enzyme_ids = tr_obj[:, 1]

        if args.aug == 0:
            aug_embedding = fp_label[tr_obj_compound_ids]
        elif args.aug == 1:
            aug_embedding = ec_label[tr_obj_enzyme_ids]
        elif args.aug == 2:
            aug_embedding = enzyme_ko_hot[tr_obj_enzyme_ids]
        elif args.aug == 3:
            aug_embedding = compound_rclass[tr_obj_compound_ids]
        elif args.aug == 5:
            aug_embedding = torch.cat([fp_label[tr_obj_compound_ids], ec_label[tr_obj_enzyme_ids],
                                       enzyme_ko_hot[tr_obj_enzyme_ids], compound_rclass[tr_obj_compound_ids]], dim=-1)
        else:
            raise NotImplementedError

        pred_interaction = model(tr_obj_compound_ids, tr_obj_enzyme_ids, aug_embedding)

        # compute loss
        # main task: compound-enzyme interaction
        loss = weighted_binary_cross_entropy(pred_interaction, tr_obj[:, -1].reshape([-1, 1]).float())

        # back propagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        if iter % args.eval_freq == 0:
            print(iter, end=', ')
            _, val_map = evaluate(model)

            if val_map > best_valid_map:
                best_valid_map = val_map
                best_model_state = copy.deepcopy(model.state_dict())

            # early stop on map
            val_maps.append(val_map)
            if len(val_maps) == args.early_stop_window // args.eval_freq:
                if val_maps[0] > np.max(val_maps[1:]):
                    break
                val_maps.pop(0)

    model.load_state_dict(best_model_state)
    test_auc, test_map = evaluate(model, report_metric_bool=True, num_compound=num_compound, num_enzyme=num_enzyme, save_result_file='NMF-BA')


def evaluate(model, report_metric_bool=False, **kwargs):
    with torch.no_grad():
        model.eval()

        batch_size = 20480
        pred_interaction = []
        for bi in range(int(np.ceil(te_pn.shape[0] / batch_size))):
            indices_s = bi * batch_size
            indices_e = min(te_pn.shape[0], (bi + 1) * batch_size)

            compound_indices = te_pn[indices_s:indices_e, 0]
            ec_indices = te_pn[indices_s:indices_e, 1]

            if args.aug == 0:
                aug_embedding = fp_label[compound_indices]
            elif args.aug == 1:
                aug_embedding = ec_label[ec_indices]
            elif args.aug == 2:
                aug_embedding = enzyme_ko_hot[ec_indices]
            elif args.aug == 3:
                aug_embedding = compound_rclass[compound_indices]
            elif args.aug == 5:
                aug_embedding = torch.cat([fp_label[compound_indices], ec_label[ec_indices],
                                           enzyme_ko_hot[ec_indices], compound_rclass[compound_indices]], dim=-1)
            else:
                raise NotImplementedError

            pred_interaction_ = model(compound_indices, ec_indices, aug_embedding)
            pred_interaction.append(pred_interaction_)
        pred_interaction = torch.cat(pred_interaction, dim=0)

        true_interaction = te_pn[:, -1].reshape([-1, 1]).float().cpu().detach().numpy().reshape(-1)
        pred_interaction = pred_interaction.cpu().detach().numpy().reshape(-1)

        if 'save_result_file' in kwargs:
            np.savez(os.path.join('./result/prediction', kwargs['save_result_file']), true=true_interaction, pred=pred_interaction)

        te_auc = roc_auc_score(y_true=true_interaction, y_score=pred_interaction)
        te_map = sk_m.average_precision_score(y_true=true_interaction, y_score=pred_interaction)

        print('auc %.3f, map %.3f' % (te_auc, te_map))

        if report_metric_bool:
            te_pn_ = te_pn.cpu().detach().numpy()

            test_rst = report_metric(kwargs['num_compound'], kwargs['num_enzyme'], true_interaction, pred_interaction, te_pn_)
            test_rst['auc'] = te_auc
            test_rst['map'] = te_map

            for key in ['auc', 'map', 'enzyme_map', 'compound_map', 'rprecision', 'enzyme_rprecision', 'compound_rprecision',
                        'enzyme_precision_1', 'compound_precision_1', 'enzyme_map_3', 'compound_map_3']:
                if isinstance(test_rst[key], tuple):
                    print('%.3f' % (test_rst[key][0]), end=' ')
                else:
                    print('%.3f' % (test_rst[key]), end=' ')
            print()

            # for key, val in test_rst.items():
            #     print('%s %.3f' % (key, val))

    return te_auc, te_map


def compute_rclass():
    compound_rclass = torch.zeros([num_compound, rpairs_pos.shape[1]-2]).to(rpairs_pos.device)

    for i in range(rpairs_pos.shape[0]):
        rpair_i = rpairs_pos[i]
        compound_rclass[rpair_i[0]] += rpair_i[2:]
        compound_rclass[rpair_i[1]] += rpair_i[2:]

    compound_rclass = torch.clamp(compound_rclass, min=0.0, max=1.0)

    return compound_rclass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--gpu', type=int, default=0, help='Weight the sum of all pos and all neg equally')

    parser.add_argument('--epoch', type=int, default=3500, help='Number of epochs.')

    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate.')
    parser.add_argument('--l2_reg', type=float, default=1e-6)
    parser.add_argument('--neg_rate', type=int, default=25)

    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')

    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--early_stop_window', type=int, default=200)

    parser.add_argument('--aug', type=int, default=0, help='0, 1, 2, 3, 5: fp, ec, ko, rpair, all')

    args = parser.parse_args()

    print(args)

    device = 'cuda:0' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'

    # load data
    tr_p, va_p, te_p, va_pn, te_pn, n_all_exclusive, num_compound, num_enzyme, compound_i2n, enzyme_i2n, fp_label, ec_label = load_data()
    rpairs_pos, _, _, _, enzyme_ko_hot, _, _ = read_mt_data(compound_i2n, enzyme_i2n)
    compound_rclass = compute_rclass()

    train()

import copy
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from util import load_interaction_data, load_mt_data, report_metric
import sklearn.metrics as sk_m
import torch
import torch.nn.functional as F


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
    def __init__(self, num_compound, num_enzyme, hidden_dim, dropout=0.5, device='cpu'):
        super(RecommenderGNN, self).__init__()

        # embedding layer for compound and enzyme
        self.MF_Embedding_Compound = torch.nn.Embedding(num_compound, hidden_dim).to(device)
        self.MF_Embedding_Enzyme = torch.nn.Embedding(num_enzyme, hidden_dim).to(device)

        self.MLP_Embedding_Compound = torch.nn.Embedding(num_compound, hidden_dim).to(device)
        self.MLP_Embedding_Enzyme = torch.nn.Embedding(num_enzyme, hidden_dim).to(device)
        self.dropout = torch.nn.Dropout(p=dropout)

        # main-task: compound-enzyme interaction prediction net. * 2 since concatenation
        self.ce_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 1),
            torch.nn.Sigmoid()
        )
        # multi-task: fingerprint
        self.fp_predictor = MLPModel(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=167, dropout=dropout, sigmoid_last_layer=True)
        # multi-task: ec
        self.ec_predictor = torch.nn.ModuleList()
        for ec_dim in [7, 68, 231]:
            self.ec_predictor.append(MLPModel(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=ec_dim, dropout=dropout, sigmoid_last_layer=False))

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU()
        )

        # save parameters
        self.num_compound = num_compound
        self.num_enzyme = num_enzyme

    def forward(self, compound_ids, enzyme_ids):
        mf_embedding_compound = self.MF_Embedding_Compound(compound_ids)
        mf_embedding_enzyme = self.MF_Embedding_Enzyme(enzyme_ids)

        mlp_embedding_compound = self.MLP_Embedding_Compound(compound_ids)
        mlp_embedding_enzyme = self.MLP_Embedding_Enzyme(enzyme_ids)

        mf_vector = mf_embedding_enzyme * mf_embedding_compound

        mlp_vector = torch.cat([mlp_embedding_enzyme, mlp_embedding_compound], dim=-1)
        mlp_vector = self.fc1(mlp_vector)

        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)

        predict_vector = self.dropout(predict_vector)

        predict_vector = self.ce_predictor(predict_vector)

        return predict_vector

    def predict_fp(self, compound_ids, mf=True):
        if mf:
            emb_compound_sub = self.MF_Embedding_Compound(compound_ids)
        else:
            emb_compound_sub = self.MLP_Embedding_Compound(compound_ids)

        pred = self.fp_predictor(emb_compound_sub)
        return pred

    def predict_ec(self, ec_ids, ec_i, mf=True):
        if mf:
            emb_enzyme_sub = self.MF_Embedding_Enzyme(ec_ids)
        else:
            emb_enzyme_sub = self.MLP_Embedding_Enzyme(ec_ids)

        pred = self.ec_predictor[ec_i](emb_enzyme_sub)
        return pred

    def triplet_loss(self, triplets, fp, margin, MF=True):
        if fp:
            embedding_anchor = self.MF_Embedding_Compound(triplets[:, 0]) if MF else self.MLP_Embedding_Compound(triplets[:, 0])
            embedding_positive = self.MF_Embedding_Compound(triplets[:, 1]) if MF else self.MLP_Embedding_Compound(triplets[:, 1])
            embedding_negative = self.MF_Embedding_Compound(triplets[:, 2]) if MF else self.MLP_Embedding_Compound(triplets[:, 2])
        else:
            embedding_anchor = self.MF_Embedding_Enzyme(triplets[:, 0]) if MF else self.MF_Embedding_Enzyme(triplets[:, 0])
            embedding_positive = self.MF_Embedding_Enzyme(triplets[:, 1]) if MF else self.MF_Embedding_Enzyme(triplets[:, 1])
            embedding_negative = self.MF_Embedding_Enzyme(triplets[:, 2]) if MF else self.MF_Embedding_Enzyme(triplets[:, 2])

        cosine_positive = F.cosine_similarity(embedding_anchor, embedding_positive, dim=-1)
        cosine_negative = F.cosine_similarity(embedding_anchor, embedding_negative, dim=-1)

        loss = torch.clamp_min(-cosine_positive + cosine_negative + margin, min=0.0)

        loss = loss.mean()

        return loss


def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output, 1e-6, 1.0 - 1e-6)

    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def contrastive_loss(pair_pos, n, fp):
    pair_pos_ = pair_pos.repeat((args.neg_rate, 1))
    pair_neg_ = np.random.choice(np.arange(n), pair_pos_.shape[0])
    pair_neg_ = torch.LongTensor(pair_neg_).to(device).unsqueeze(-1)
    pair_ = torch.cat((pair_pos_, pair_neg_), dim=-1)
    loss_pair_mf = model.triplet_loss(pair_, fp=fp, margin=args.margin, MF=True)
    loss_pair_mlp = model.triplet_loss(pair_, fp=fp, margin=args.margin, MF=False)

    loss = loss_pair_mf + loss_pair_mlp

    return loss


def train():
    val_maps = []
    best_valid_map = 0.0
    best_model_state = None

    # train
    for t in (range(args.iterations)):
        model.train()

        # compute interaction loss
        tr_p_obj = tr_p
        # sample negative links
        tr_n_ids = np.random.choice(np.arange(n_all_exclusive.shape[0]), tr_p_obj.shape[0] * args.neg_rate)
        tr_n_obj = n_all_exclusive[tr_n_ids]
        tr_obj = torch.cat([tr_p_obj, tr_n_obj], dim=0)
        tr_obj_compound_ids = tr_obj[:, 0]
        tr_obj_enzyme_ids = tr_obj[:, 1]
        # forward and compute loss
        pred_interaction = model(tr_obj_compound_ids, tr_obj_enzyme_ids)
        loss_interaction = weighted_binary_cross_entropy(pred_interaction, tr_obj[:, -1].reshape([-1, 1]).float())

        # compute multi-task loss
        # multi-task: fingerprints
        loss_fp_mf = weighted_binary_cross_entropy(model.predict_fp(torch.arange(num_compound).to(device), mf=True), fp_label)
        loss_fp_mlp = weighted_binary_cross_entropy(model.predict_fp(torch.arange(num_compound).to(device), mf=False), fp_label)
        loss_fp = loss_fp_mf + loss_fp_mlp

        # multi-task: ec
        loss_ec_mf, loss_ec_mlp = 0.0, 0.0
        ec_loss_w = [1./3., 1./3., 1./3.]
        for j, ec_dim in enumerate([(0, 7), (7, 7+68), (7+68, 7+68+231)]):
            ec_indices = torch.arange(num_enzyme).to(device)
            ec_label_j = ec_label[:, ec_dim[0]:ec_dim[1]]
            _, ec_label_j = ec_label_j.max(dim=1)
            loss_ec_mf += ec_loss_w[j] * torch.nn.CrossEntropyLoss()(model.predict_ec(ec_indices, j, mf=True), ec_label_j)
            loss_ec_mlp += ec_loss_w[j] * torch.nn.CrossEntropyLoss()(model.predict_ec(ec_indices, j, mf=False), ec_label_j)
        loss_ec = loss_ec_mf + loss_ec_mlp

        # multi-task: rpair
        loss_rpair = contrastive_loss(rpairs_pos[:, :2], num_compound, fp=True)

        # multi-task: ko
        loss_enzyme_ko = contrastive_loss(enzyme_ko, num_enzyme, fp=False)

        # compute training loss with dynamic weighting
        T = 2000
        w_m = 1.0 if t > T else t / float(T)
        w_a = 0.0 if t > T else (1 - t / float(T))
        loss = w_m * loss_interaction + w_a * (loss_fp + loss_ec + loss_rpair + loss_enzyme_ko)

        # back propagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        if t % args.eval_freq == 0 or t == args.iterations - 1:
            _, val_map = evaluate(model, va_pn, iteration=t)
            if val_map > best_valid_map:
                best_valid_map = val_map
                best_model_state = copy.deepcopy(model.state_dict())

            # early stop on auc
            val_maps.append(val_map)
            if len(val_maps) == args.early_stop_window // args.eval_freq:
                if val_maps[0] > np.max(val_maps[1:]):
                    break
                val_maps.pop(0)

    # testing
    model.load_state_dict(best_model_state)

    evaluate(model, te_pn, report_metric_bool=True, iteration=-1, num_compound=num_compound, num_enzyme=num_enzyme)


def evaluate(model, pn_, report_metric_bool=False, **kwargs):
    with torch.no_grad():
        model.eval()

        # forward
        batch_size = 20480
        pred_interaction = []
        for bi in range(int(np.ceil(pn_.shape[0] / batch_size))):
            indices_s = bi * batch_size
            indices_e = min(pn_.shape[0], (bi + 1) * batch_size)
            compound_indices = pn_[indices_s:indices_e, 0]
            ec_indices = pn_[indices_s:indices_e, 1]
            pred_interaction_ = model(compound_indices, ec_indices)
            pred_interaction.append(pred_interaction_)
        pred_interaction = torch.cat(pred_interaction, dim=0)
        # convert ground truth and prediction to numpy
        true_interaction = pn_[:, -1].reshape([-1, 1]).float().cpu().detach().numpy().reshape(-1)
        pred_interaction = pred_interaction.cpu().detach().numpy().reshape(-1)

        # report metrics for evaluation
        te_auc = roc_auc_score(y_true=true_interaction, y_score=pred_interaction)
        te_map = sk_m.average_precision_score(y_true=true_interaction, y_score=pred_interaction)

        print('Iteration at %d: auc %.3f, map %.3f' % (kwargs['iteration'], te_auc, te_map))

        if report_metric_bool:
            test_rst = report_metric(kwargs['num_compound'], kwargs['num_enzyme'], true_interaction, pred_interaction, pn_.cpu().detach().numpy())
            test_rst['auc'] = te_auc
            test_rst['map'] = te_map

            for key in ['auc', 'map', 'enzyme_map', 'compound_map', 'rprecision', 'enzyme_rprecision', 'compound_rprecision',
                        'enzyme_precision_1', 'compound_precision_1', 'enzyme_map_3', 'compound_map_3']:
                if isinstance(test_rst[key], tuple):
                    print('%.3f' % (test_rst[key][0]), end=' ')
                else:
                    print('%.3f' % (test_rst[key]), end=' ')
            print()

    return te_auc, te_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLT with NMF.")
    parser.add_argument('--gpu', type=int, default=0)
    # training parameters
    parser.add_argument('--iterations', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--l2_reg', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--neg_rate', type=int, default=1)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--early_stop_window', type=int, default=200)
    # model structure
    parser.add_argument('--hidden_dim', type=int, default=512)
    args = parser.parse_args()
    print(args)

    device = 'cuda:0' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'

    # load data
    tr_p, va_p, te_p, va_pn, te_pn, n_all_exclusive, num_compound, num_enzyme, compound_i2n, enzyme_i2n, fp_label, ec_label = load_interaction_data()
    rpairs_pos, cpd_module, cpd_pathway, enzyme_ko, enzyme_ko_hot, enzyme_module, enzyme_pathway = load_mt_data()

    # construct model
    model = RecommenderGNN(num_compound=num_compound, num_enzyme=num_enzyme,
                           hidden_dim=args.hidden_dim, dropout=args.dropout, device=device).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)

    # start training
    train()

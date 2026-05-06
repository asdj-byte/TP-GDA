import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.autograd import Function
from DataLoader import *
import os
import warnings


def load_data(uarch, power_group):
    feat_list = []
    for idx in range(len(comp)):
        comp_name = comp[idx]
        loaded_feat = np.load(f'../dataset/component_feature/{comp_name}.npy')
        feat_list.append(loaded_feat)
    feature = np.hstack(feat_list)
    label_path = '../dataset/label.npy'
    if not os.path.exists(label_path): label_path = f'../dataset/{uarch}_label.npy'
    label = np.load(label_path)
    label = label.reshape((label.shape[0], 12, 5))
    label = label[:, :, power_group]
    if uarch == 'BOOM':
        return feature[0:120], label[0:120]
    else:
        return feature[120:], label[120:]


def generate_training_test(feature, label_total, training_index, testing_index):
    training_set = [idx * 8 + i for idx in training_index for i in range(8)]
    testing_set = [idx * 8 + i for idx in testing_index for i in range(8)]
    return feature[training_set], label_total[training_set], feature[testing_set], label_total[testing_set]



logic_bias = 0
dtlb_bias = 0


def compute_reserve_station_entries(decodewidth_init):
    decodewidth = int(decodewidth_init + 0.01)
    idx = min(max(0, decodewidth - 1), 4)
    isu_params = [[8, decodewidth] * 3, [12, decodewidth, 20, decodewidth, 16, decodewidth],
                  [16, decodewidth, 32, decodewidth, 24, decodewidth],
                  [24, decodewidth, 40, decodewidth, 32, decodewidth],
                  [24, decodewidth, 40, decodewidth, 32, decodewidth]]
    p = isu_params[idx]
    return p[0] + p[2] + p[4]


def estimate_bias_logic(feature, label):
    global logic_bias
    if len(feature) == 0: return
    reg = LinearRegression().fit(feature.reshape(-1, 1), label.reshape(-1, 1))
    bias = reg.intercept_[0]
    alpha = reg.coef_[0][0]
    logic_bias = bias / (alpha + 1e-9)


def estimate_bias_dtlb(feature, label):
    global dtlb_bias
    if len(feature) == 0: return
    reg = LinearRegression().fit(feature.reshape(-1, 1), label.reshape(-1, 1))
    bias = reg.intercept_[0]
    alpha = reg.coef_[0][0]
    dtlb_bias = bias / (alpha + 1e-9)


def encode_arch_knowledge(component_name, feature, label, is_training=False):
    eps = 1e-9
    if component_name in ["BP", "ICache", "DCache", "RNU", "ROB", "IFU", "LSU"]:
        scale = np.prod(feature[:, encode_table[component_name]], axis=1)
        return label / (scale + eps)
    elif component_name == "Regfile":
        scale = np.sum(feature[:, encode_table[component_name]], axis=1)
        return label / (scale + eps)
    elif component_name == "ISU":
        dw = feature[:, encode_table[component_name][0]]
        rs = np.array([compute_reserve_station_entries(w) for w in dw])
        return label / (rs + eps)
    elif component_name == "Others":
        idx = encode_table[component_name][0]
        if is_training: estimate_bias_logic(feature[:, idx], label)
        return label / (feature[:, idx] + logic_bias + eps)
    elif component_name == "D-TLB":
        idx = encode_table[component_name][0]
        if is_training: estimate_bias_dtlb(feature[:, idx], label)
        return label / (feature[:, idx] + dtlb_bias + eps)
    return label


def decode_arch_knowledge(component_name, feature, pred):
    if component_name in ["BP", "ICache", "DCache", "RNU", "ROB", "IFU", "LSU"]:
        scale = np.prod(feature[:, encode_table[component_name]], axis=1)
        return pred * scale
    elif component_name == "Regfile":
        scale = np.sum(feature[:, encode_table[component_name]], axis=1)
        return pred * scale
    elif component_name == "ISU":
        dw = feature[:, encode_table[component_name][0]]
        rs = np.array([compute_reserve_station_entries(w) for w in dw])
        return pred * rs
    elif component_name == "Others":
        return pred * (feature[:, encode_table[component_name][0]] + logic_bias)
    elif component_name == "D-TLB":
        return pred * (feature[:, encode_table[component_name][0]] + dtlb_bias)
    return pred



class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None



class DANN_Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(DANN_Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, alpha):
        x = ReverseLayerF.apply(x, alpha)
        return self.net(x)


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.2, alpha=0.2, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        B, N, _ = h.size()
        q = self.W(h).view(B, N, self.num_heads, self.out_dim)
        q_repeat = q.unsqueeze(2).repeat(1, 1, N, 1, 1)
        k_repeat = q.unsqueeze(1).repeat(1, N, 1, 1, 1)
        combined = torch.cat([q_repeat, k_repeat], dim=-1)
        e = self.leakyrelu(self.a(combined)).squeeze(-1)
        mask = (adj > 0).float().view(1, N, N, 1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        v = q.unsqueeze(1)
        h_prime = torch.sum(attention.unsqueeze(-1) * v, dim=2)

        if self.concat:
            return F.elu(h_prime.view(B, N, self.num_heads * self.out_dim))
        else:
            return h_prime.mean(dim=2)


class EnhancedPANDA_GAT(nn.Module):
    def __init__(self, feature_dims, hidden_dim=32, num_heads=4, num_nodes=11):
        super(EnhancedPANDA_GAT, self).__init__()

        self.input_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1)
            ) for dim in feature_dims
        ])

        self.node_type_emb = nn.Embedding(num_nodes, hidden_dim)

        self.gat1 = MultiHeadGATLayer(hidden_dim, hidden_dim, num_heads=num_heads, concat=True)
        self.gat2 = MultiHeadGATLayer(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, concat=False)

        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ) for _ in range(len(feature_dims))
        ])

    def forward(self, features_list, adj):
        emb, pred = self.get_embeddings_and_pred(features_list, adj)
        return pred.squeeze(-1)

    def get_embeddings_and_pred(self, features_list, adj):
        B = features_list[0].shape[0]
        device = features_list[0].device
        encoded_feats = [enc(f) for i, (enc, f) in enumerate(zip(self.input_encoders, features_list))]
        x_phys = torch.stack(encoded_feats, dim=1)
        node_ids = torch.arange(len(features_list)).to(device)
        x_type = self.node_type_emb(node_ids).unsqueeze(0).expand(B, -1, -1)
        x = x_phys + x_type

        x = self.gat1(x, adj)
        embedding = self.gat2(x, adj)

        preds = []
        for i, decoder in enumerate(self.output_layers):
            preds.append(decoder(embedding[:, i, :]))
        pred_tensor = torch.cat(preds, dim=1).unsqueeze(2)

        return embedding, pred_tensor


def get_cpu_adj(num_nodes):
    adj = torch.eye(num_nodes)
    conns = [(0, 2), (1, 2), (2, 3), (3, 8), (3, 7), (7, 6), (7, 9), (7, 4), (4, 5), (10, 8), (10, 3)]
    for s, d in conns: adj[s, d] = adj[d, s] = 1.0
    return adj




def train_gat_pretraining(tr_feature, tr_label, te_feature, epochs=1000, lr=0.002):

    feats_np_s, dims, scalers, lbl_enc = [], [], [], np.zeros((tr_label.shape[0], 11))


    for i, c in enumerate(comp):
        s, e = feature_of_components[c]
        raw = tr_feature[:, s:e]
        scaler = SklearnScaler()
        feats_np_s.append(scaler.fit_transform(raw))
        scalers.append(scaler)
        dims.append(e - s)
        lbl_enc[:, i] = encode_arch_knowledge(c, raw, tr_label[:, i + 1], True)

    feats_np_t = []
    for i, c in enumerate(comp):
        s, e = feature_of_components[c]
        raw = te_feature[:, s:e]
        feats_np_t.append(scalers[i].transform(raw))


    lbl_enc_log = np.log1p(lbl_enc)

    tgt_scaler = SklearnScaler()  
    y_log_scaled = torch.FloatTensor(tgt_scaler.fit_transform(lbl_enc_log))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedPANDA_GAT(dims, hidden_dim=32, num_heads=4).to(device)
    discriminator = DANN_Discriminator(input_dim=32).to(device)  # 每个节点 Embedding dim=32

    opt = optim.Adam(list(model.parameters()) + list(discriminator.parameters()), lr=lr, weight_decay=1e-4)
    loss_reg_fn = nn.MSELoss() 
    loss_dom_fn = nn.BCELoss() 

    adj = get_cpu_adj(len(comp)).to(device)
    feats_s = [torch.FloatTensor(f).to(device) for f in feats_np_s]
    feats_t = [torch.FloatTensor(f).to(device) for f in feats_np_t]
    y = y_log_scaled.to(device)

    model.train()
    discriminator.train()

    min_len = min(feats_s[0].shape[0], feats_t[0].shape[0])

    for epoch in range(epochs):

        p = float(epoch) / epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        opt.zero_grad()


        s_emb, s_pred = model.get_embeddings_and_pred([f[:min_len] for f in feats_s], adj)
        t_emb, _ = model.get_embeddings_and_pred([f[:min_len] for f in feats_t], adj)


        loss_reg = loss_reg_fn(s_pred.squeeze(-1), y[:min_len])


        s_dom_out = discriminator(s_emb.view(-1, 32), alpha)
        t_dom_out = discriminator(t_emb.view(-1, 32), alpha)

        s_target = torch.zeros(s_dom_out.size(0), 1).to(device)  # Source = 0
        t_target = torch.ones(t_dom_out.size(0), 1).to(device)  # Target = 1

        loss_dom = loss_dom_fn(s_dom_out, s_target) + loss_dom_fn(t_dom_out, t_target)

        loss = loss_reg + 0.1 * loss_dom  
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
        opt.step()

        if epoch % 200 == 0:
            print(f"      Epoch {epoch}: Reg Loss={loss_reg.item():.4f}, Dom Loss={loss_dom.item():.4f}")

    return model, scalers, tgt_scaler


def train_xgb_hybrid(gnn_model, feature, label, scalers, tgt_scaler, current_seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model.eval()
    adj = get_cpu_adj(len(comp)).to(device)

    feats_t = [
        torch.FloatTensor(scalers[i].transform(feature[:, feature_of_components[c][0]:feature_of_components[c][1]])).to(
            device) for i, c in enumerate(comp)]

    with torch.no_grad():
        emb, pred = gnn_model.get_embeddings_and_pred(feats_t, adj)
        emb = emb.cpu().numpy()
        pred = pred.squeeze(2).cpu().numpy()
       


    pcas = []
    emb_reduced_list = []
    for i in range(len(comp)):
        n_comp = min(4, emb.shape[2], emb.shape[0])
        pca = PCA(n_components=n_comp, random_state=current_seed)
        emb_red = pca.fit_transform(emb[:, i, :])
        pcas.append(pca)
        emb_reduced_list.append(emb_red)

    models = []
    for i, c in enumerate(comp):
        s, e = feature_of_components[c]
        raw = feature[:, s:e]


        gnn_feat = emb_reduced_list[i]
        gnn_val = pred[:, i:i + 1] 
        X = np.hstack([raw, gnn_feat, gnn_val])

       
        y_phys = encode_arch_knowledge(c, raw, label[:, i + 1], True)
        y_log = np.log1p(y_phys)

        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=1,
            random_state=current_seed,
            objective='reg:squarederror'
        )
        xgb_model.fit(X, y_log)
        models.append(xgb_model)

    return models, pcas


def test_hybrid(gnn_model, models, pcas, feature, label, scalers, tgt_scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model.eval()
    adj = get_cpu_adj(len(comp)).to(device)

    feats_t = [
        torch.FloatTensor(scalers[i].transform(feature[:, feature_of_components[c][0]:feature_of_components[c][1]])).to(
            device) for i, c in enumerate(comp)]

    with torch.no_grad():
        emb, pred = gnn_model.get_embeddings_and_pred(feats_t, adj)
        emb = emb.cpu().numpy()
        pred = pred.squeeze(2).cpu().numpy()

    preds = []
    for i, c in enumerate(comp):
        s, e = feature_of_components[c]
        raw = feature[:, s:e]

        gnn_feat = pcas[i].transform(emb[:, i, :])
        gnn_val = pred[:, i:i + 1]
        X = np.hstack([raw, gnn_feat, gnn_val])


        p_log = models[i].predict(X)


        p_phys = np.expm1(p_log)

        p_phys = np.maximum(p_phys, 0)

        preds.append(decode_arch_knowledge(c, raw, p_phys))

    res = np.vstack(preds).T
    return np.hstack([np.sum(res, axis=1, keepdims=True), res])



def run_ensemble(tr_idx, te_idx, uarch, name, seeds=[42, 2024, 0, 123, 999]):
    global logic_bias, dtlb_bias
    logic_bias = dtlb_bias = 0

    feat, lbl = load_data(uarch, 0)
    tr_x, tr_y, te_x, te_y = generate_training_test(feat, lbl, tr_idx, te_idx)

    print(f"\n>>> Running Ensemble PANDA (GAT+DANN+XGBoost, {len(seeds)} Seeds) on {uarch}-{name}")

    ensemble_preds = []

    for i, seed in enumerate(seeds):
        print(f"   [Run {i + 1}/{len(seeds)}] Seed={seed} ...")
        setup_seed(seed)


        gnn, scalers, tgt_scaler = train_gat_pretraining(tr_x, tr_y, te_x, epochs=1200, lr=0.002)

        xgbs, pcas = train_xgb_hybrid(gnn, tr_x, tr_y, scalers, tgt_scaler, current_seed=seed)


        pred = test_hybrid(gnn, xgbs, pcas, te_x, te_y, scalers, tgt_scaler)
        ensemble_preds.append(pred)

    print("   [Ensemble] Averaging predictions...")
    avg_preds = np.mean(np.array(ensemble_preds), axis=0)

    if not os.path.exists("PANDA_XGB_GAT_Ensemble"): os.makedirs("PANDA_XGB_GAT_Ensemble")
    for i in range(12):
        draw_figure(te_y[:, i], avg_preds[:, i], f"{uarch}_{name}_{figure_name[i]}")


if __name__ == "__main__":
    if not os.path.exists("PANDA_XGB_GAT_Ensemble"): os.makedirs("PANDA_XGB_GAT_Ensemble")

    # BOOM Cases
    run_ensemble([0, 7, 14], [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13], "BOOM", "evenly")
    run_ensemble([0, 1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "BOOM", "small")
    run_ensemble([12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "BOOM", "large")

    # XS Cases (The problematic ones)
    run_ensemble([0, 5, 9], [1, 2, 3, 4, 6, 7, 8], "XS", "evenly")
    run_ensemble([0, 1, 2], [3, 4, 5, 6, 7, 8, 9], "XS", "small")
    run_ensemble([7, 8, 9], [0, 1, 2, 3, 4, 5, 6], "XS", "large")

import argparse
import distutils.util
import os
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch as th
from models_gcn import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
dev = th.device('cuda' if th.cuda.is_available() else 'cpu')


def adj_preprocess(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(axis=1).A1
    deg = sp.diags(rowsum ** (-0.5))
    adj_ = deg @ adj_ @ deg.tocsr()

    return adj_


def buildtensor(adj):
    sparserow = th.LongTensor(adj.row).unsqueeze(1)
    sparsecol = th.LongTensor(adj.col).unsqueeze(1)
    sparseconcat = th.cat((sparserow, sparsecol), 1).cuda()
    sparsedata = th.FloatTensor(adj.data).cuda()
    adjtensor = th.sparse.FloatTensor(sparseconcat.t(), sparsedata, th.Size(adj.shape)).cuda()

    return adjtensor


def compute_acc(pred, labels, mask=None):
    if mask is None:
        return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)
    else:
        return (th.argmax(pred[mask], dim=1) == labels[mask[:len(labels)]]).float().sum() / np.sum(mask)


class Dataset(object):
    def __init__(self, adj_path, feat_path, label_path, test_size=50000, indices=None):
        fg = open(adj_path, 'rb')
        self.adj = pkl.load(fg)
        self.features = np.load(feat_path)
        self.labels = np.load(label_path)

        self.num_labels = max(self.labels) + 1
        size_raw = self.features.shape[0]
        size_reduced = size_raw - test_size

        if indices is None:
            indices_train = np.array([i for i in range(size_reduced - test_size)])
            indices_val = np.array([i for i in range(size_reduced - test_size, size_reduced)])
            indices_test = np.array([i for i in range(size_raw - test_size, size_raw)])
        else:
            indices_train, indices_val, indices_test = indices

        self.train_mask = np.zeros(size_reduced).astype(bool)
        self.val_mask = np.zeros(size_reduced).astype(bool)
        self.test_mask = np.zeros(size_raw).astype(bool)
        self.train_mask[indices_train] = True
        self.val_mask[indices_val] = True
        self.test_mask[indices_test] = True


def get_noise_list(adj, K, target_noise, noise_tmp_list):
    i = 1
    res = []
    while len(res) < K and i < len(noise_tmp_list):
        if adj[target_noise, noise_tmp_list[i]] == 0:
            res.append(noise_tmp_list[i])
        i += 1

    return res


def update_noise_active(noise_active, noise_edge, threshold=100):
    for node in noise_active:
        if noise_edge[node] >= threshold:
            noise_active.pop(noise_active.index(node))
    return noise_active


def connect(target_node, num_test=50000, num_add=500, max_connection=90, num_multi=50, mode='multi-layer'):
    adj = np.zeros((num_add, num_test + num_add))
    N = len(target_node)

    if mode == 'random-inter':
        # test_node_list: a list of test nodes to be connected
        noise_edge = np.zeros(num_add)
        noise_active = [i for i in range(num_add)]

        # create edges between noise node and test node
        for i in range(N):
            if not noise_active:
                break
            noise_list = np.random.choice(noise_active, 1)
            noise_edge[noise_list] += 1
            noise_active = update_noise_active(noise_active, noise_edge)
            adj[noise_list, target_node[i]] = 1

        # create edges between noise nodes
        for i in range(len(noise_active)):
            if not noise_active:
                break
            noise_tmp_list = sorted(noise_active, key=lambda x: noise_edge[x])
            target_noise = noise_tmp_list[0]
            K = max_connection - noise_edge[target_noise]
            noise_list = get_noise_list(adj, K, target_noise, noise_tmp_list)

            noise_edge[noise_list] += 1
            noise_edge[target_noise] += len(noise_list)

            noise_active = update_noise_active(noise_active, noise_edge)
            if noise_list:
                adj[target_noise, num_test + np.array(noise_list)] = 1
                adj[noise_list, num_test + target_noise] = 1

    elif mode == 'multi-layer':
        # test_node_list: a list of test nodes to be connected
        noise_edge = np.zeros(num_add)
        noise_active = [i for i in range(num_add - num_multi)]

        # create edges between noise node and test node
        for i in range(N):
            if not noise_active:
                break
            noise_list = np.random.choice(noise_active, 1)
            noise_edge[noise_list] += 1
            noise_active = update_noise_active(
                noise_active, noise_edge, threshold=max_connection)
            adj[noise_list, target_node[i]] = 1

        # create edges between noise nodes
        for i in range(len(noise_active)):
            if not noise_active:
                break
            noise_tmp_list = sorted(noise_active, key=lambda x: noise_edge[x])
            target_noise = noise_tmp_list[0]
            K = max_connection - noise_edge[target_noise]
            noise_list = get_noise_list(adj, K, target_noise, noise_tmp_list)

            noise_edge[noise_list] += 1
            noise_edge[target_noise] += len(noise_list)

            noise_active = update_noise_active(
                noise_active, noise_edge, threshold=max_connection)

            if noise_list:
                adj[target_noise, num_test + np.array(noise_list)] = 1
                adj[noise_list, num_test + target_noise] = 1

        noise_active_layer2 = [i for i in range(num_multi)]
        noise_edge_layer2 = np.zeros(num_multi)
        for i in range(num_add - num_multi):
            if not noise_active_layer2:
                break
            noise_list = np.random.choice(noise_active_layer2, 10)
            noise_edge_layer2[noise_list] += 1
            noise_active_layer2 = update_noise_active(
                noise_active_layer2, noise_edge_layer2, threshold=max_connection)
            adj[noise_list + num_add - num_multi, i + num_test] = 1
            adj[i, noise_list + num_test + num_add - num_multi] = 1

    else:
        print("Mode ERROR: 'mode' should be one of ['random-inter', 'multi-layer']")

    return adj


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("speit attack")

    # Model parameters
    argparser.add_argument('--data-dir', type=str, default='dset')
    argparser.add_argument('--target-path', type=str, default='target_node_0726.npy')
    argparser.add_argument('--save-path', type=str, default='result')

    argparser.add_argument('--model', type=str, default='gcn_lm')
    argparser.add_argument('--adj-norm', type=lambda x: bool(distutils.util.strtobool(x)), default=True)
    argparser.add_argument('--feat-norm', type=str, default=None)

    argparser.add_argument('--num-test', type=int, default=50000)
    argparser.add_argument('--num-add', type=float, default=500)
    argparser.add_argument('--max-connections', type=int, default=89)
    argparser.add_argument('--num-multi', type=int, default=50)

    # Attack parameters
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--lr', type=float, default=0.1)
    argparser.add_argument('--feature-limit', type=float, default=2.0)

    args = argparser.parse_args()

    # Load data
    DIR_DATA = args.data_dir
    adj_path = os.path.join(DIR_DATA, "testset_adj.pkl")
    feat_path = os.path.join(DIR_DATA, "features.npy")
    label_path = os.path.join(DIR_DATA, "label_a.npy")
    dataset = Dataset(adj_path, feat_path, label_path)
    adj = dataset.adj
    features = dataset.features
    labels = dataset.labels
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    size_raw = features.shape[0]
    size_reduced = size_raw - args.num_test
    num_features = features.shape[1]

    # Load model
    model = []
    exec("model.append(" + args.model + "(100,18).cuda())")
    model_path = args.model + '_aminer/0'
    model_states = th.load(model_path, map_location=dev)
    model[-1].load_state_dict(model_states)
    model = model[-1].to(dev)
    model.eval()

    # prediction on raw graph (without attack nodes)
    features = th.FloatTensor(features).to(dev)
    labels = th.LongTensor(labels).to(dev)
    if args.adj_norm:
        adj = adj_preprocess(adj)
    adj_tensor = buildtensor(adj.tocoo())
    pred_raw = model(features, adj_tensor)

    # select the least probable class as the target class
    pred_raw_label = th.argmax(pred_raw[:size_raw][test_mask], 1)
    pred_test_prob = th.softmax(pred_raw[:size_raw][test_mask], 1)
    attack_label = th.argsort(pred_test_prob, 1)[:, 2]

    # Generate attack matrix (with target nodes to be attacked)
    #target_node = np.load(args.target_path)
    print(len(features))
    target_node=[]
    for i in range(args.num_test):
        target_node.append(i)
    target_node=np.array(target_node)
    adj_attack = connect(target_node, args.num_test, args.num_add,
                         args.max_connections, args.num_multi, mode='multi-layer')
    adj_attack = sp.csr_matrix(adj_attack)
    adj_adv = sp.hstack([sp.csr_matrix(np.zeros([args.num_add, size_raw - args.num_test])), adj_attack])
    adj_adv = sp.csr_matrix(adj_adv)
    adj_adv_ = sp.vstack([adj, adj_adv[:, :size_raw]])
    adj_adv = sp.hstack([adj_adv_, adj_adv.T])
    if args.adj_norm:
        adj_adv = adj_preprocess(adj_adv)
    adj_adv_tensor = buildtensor(adj_adv.tocoo())

    feat_ae = np.zeros((args.num_add, features.shape[1]))
    features_ae = th.FloatTensor(feat_ae).to(dev)
    features_ae.requires_grad_(True)

    # Optimizer
    # optimizer = th.optim.Adam([features_ae], lr=args.lr)
    optimizer = th.optim.Adadelta([features_ae], lr=100 * args.lr, rho=0.9, eps=1e-06, weight_decay=0)
    # optimizer = th.optim.Adagrad([features_ae], lr=args.lr, lr_decay=0,
    # weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    # optimizer = th.optim.SGD([features_ae], lr=1)

    # Gradient attack_old on features
    print(model)
    epoch = args.num_epochs
    for i in range(epoch):
        features_concat = th.cat((features, features_ae), 0)
        pred_ae = model(features_concat, adj_adv_tensor, dropout=0)
        pred_loss_0 = -F.nll_loss(pred_ae[:size_raw][test_mask], pred_raw_label).cpu()
        pred_ae_prob = th.softmax(pred_ae[:size_raw][test_mask], 1).cpu()
        pred_loss = (pred_ae_prob[[np.arange(args.num_test), pred_raw_label]] - pred_ae_prob[
            [np.arange(args.num_test), attack_label]]).sum()  + 1000 * pred_loss_0

        optimizer.zero_grad()
        pred_loss.backward(retain_graph=True)
        optimizer.step()

        with th.no_grad():
            features_ae.clamp_(-args.feature_limit, args.feature_limit)
        print("Epoch {}, Loss: {:.5f}, Loss0: {:.5f}, Test acc: {:.5f}".format(i, pred_loss,pred_loss_0, compute_acc(pred_ae[:size_raw][test_mask],
                                                                                          pred_raw_label)))

    # Show results
    print('*' * 30, "AE graph inference", '*' * 30)
    print("Feature range [{:.2f}, {:.2f}]".format(features_ae.min(), features_ae.max()))
    # On train set (493486)
    print("Acc on train: {:.4f}".format(compute_acc(pred_ae[:size_reduced], labels[:size_reduced], train_mask)))
    # On val set (50000)
    print("Acc on val: {:.4f}".format(compute_acc(pred_ae[:size_reduced], labels[:size_reduced], val_mask)))
    # On test set (50000)
    print("Acc on test: {:.4f}".format(compute_acc(pred_ae[:size_raw], labels[:size_raw], test_mask)))

    # save adversarial adjacent matrix and adversarial features
    with open(os.path.join(args.save_path, "adj.pkl"), "wb") as f:
        pkl.dump(adj_adv[-args.num_add:], f)
    np.save(os.path.join(args.save_path, "feature.npy"), features_ae.detach().cpu().numpy())

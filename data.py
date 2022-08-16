"""
Code partially copied from 'Diffusion Improves Graph Learning' repo https://github.com/klicperajo/gdc/blob/master/data.py
"""

import os

import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WikiCS
# from graph_rewiring import get_two_hop, apply_gdc
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from sklearn.model_selection import ShuffleSplit
# from graph_rewiring import make_symmetric, apply_pos_dist_rewire
from heterophilic import WebKB, WikipediaNetwork, Actor
from torch_geometric.utils import to_scipy_sparse_matrix
from utils import ROOT_DIR, generate_sparse_adj

DATA_PATH = f'{ROOT_DIR}/data'


# def rewire(data, opt, data_dir):
#   rw = opt['rewiring']
#   if rw == 'two_hop':
#     data = get_two_hop(data)
#   elif rw == 'gdc':
#     data = apply_gdc(data, opt)
#   elif rw == 'pos_enc_knn':
#     data = apply_pos_dist_rewire(data, opt, data_dir)
#   return data


def get_dataset(opt: dict, data_dir, use_lcc: bool = False) -> InMemoryDataset:
    ds = opt['dataset']
    path = os.path.join(data_dir, ds)
    if ds in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path, ds)
    elif ds in ['Computers', 'Photo']:
        dataset = Amazon(path, ds)
    elif ds == 'CoauthorCS':
        dataset = Coauthor(path, 'CS')
    elif ds in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=path, name=ds, transform=T.NormalizeFeatures())
    elif ds in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(
            root=path, name=ds, transform=T.NormalizeFeatures())
    elif ds == 'film':
        dataset = Actor(root=path, transform=T.NormalizeFeatures())
    elif ds == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=ds, root=path,
                                         transform=T.ToSparseTensor())
        use_lcc = False  # never need to calculate the lcc with ogb datasets
    # confirm with Ali once
    elif ds == 'WikiCS':
        dataset = WikiCS(root=path)
    else:
        raise Exception('Unknown dataset.')

    if use_lcc:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data
    # if opt['rewiring'] is not None:
    #   dataset.data = rewire(dataset.data, opt, data_dir)
    train_mask_exists = True
    try:
        dataset.data.train_mask
    except AttributeError:
        train_mask_exists = False

    if ds == 'ogbn-arxiv':
        split_idx = dataset.get_idx_split()
        ei = to_undirected(dataset.data.edge_index)
        data = Data(
            x=dataset.data.x,
            edge_index=ei,
            y=dataset.data.y,
            train_mask=split_idx['train'],
            test_mask=split_idx['test'],
            val_mask=split_idx['valid'])
        dataset.data = data
        train_mask_exists = True

    # Add the sparse tensor represnattion of the
    adj = to_scipy_sparse_matrix(dataset.data.edge_index)
    data.adj_t = generate_sparse_adj(adj)
    data.adj_t = data.adj_t.to_symmetric()
    dataset.data = data

    # todo this currently breaks with heterophilic datasets if you don't pass --geom_gcn_splits
    if (use_lcc or not train_mask_exists) and not opt['geom_gcn_splits'] and opt['split_policy'] == 'fixed':
        dataset.data = set_fixed_train_val_test_split(
            12345,
            dataset.data,
            num_development=5000 if ds == "CoauthorCS" else 1500)
    if (use_lcc or not train_mask_exists) and not opt['geom_gcn_splits'] and opt['split_policy'] == 'ratio':
        dataset.data = set_ratio_train_valid_test_split(
            dataset.data, opt['train_ratio'], opt['num_splits'], opt['global_random_seed'])

    return dataset


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [
            n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def set_fixed_train_val_test_split(
        seed: int,
        data: Data,
        num_development: int = 1500,
        num_per_class: int = 20) -> Data:
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(
        num_nodes, num_development, replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(
            data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(
            class_idx, num_per_class, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data


def treat_unique_class(y, dict_rep_set):
    """
    This function ensures the training data set has at least 1 sample per class.
    Input:
        y: label, could be a 0,1 matrix (number of samples x number of classes)
    Output:
        rep_set: representative indices of all samples (1 per class)
        rest_set: indices not part of rep_set
    Note: When using ShuffleSplit, only shuffle the indices not part of rep_set because
            rep_set will always be a part of training data
        Use the following code to find the rest of the indices that is not part of rep_set

    """
    ids = {}
    u, ui, uc = np.unique(y, return_counts=True, return_index=True)
    for i in range(len(dict_rep_set.keys())):
        rep_set = dict_rep_set[i]
        rest_set = np.delete(range(len(y)), rep_set)
        dict_rep = {"rep_set": rep_set, "rest_set": rest_set}
        ids[i] = dict_rep
    return ids


def set_ratio_train_valid_test_split(data: Data, train_ratio: float = 0.32, n_splits: int = 1, random_state: int = 0) -> Data:

    global_random_state = np.random.RandomState(random_state)
    splits_seeds_lst = global_random_state.randint(0, 1000000, n_splits)
    dict_Split_idx = {}

    y = data.y.numpy()
    num_nodes = data.y.shape[0]

    uniq_classes = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_classes}
    groupby_classes = {}
    for ii, level in enumerate(uniq_classes):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_classes[level] = obs_idx

    dict_rep_set = {}
    for i in range(n_splits):
        rep_set = []
        for key, values in groupby_classes.items():
            rnd_state = np.random.RandomState(splits_seeds_lst[i])
            rep = rnd_state.choice(values, 1, replace=False)
            rep_set.append(rep[0])
        dict_rep_set[i] = rep_set

    ids = treat_unique_class(y, dict_rep_set)

    index = 0
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,
                           random_state=random_state)

    for train_i, test_i in shuffle.split(y[ids[index]['rest_set']]):
        list_split_idx_3 = []
        stratSplit = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in stratSplit.split(ids[index]['rest_set'][train_i]):
            X_train = ids[index]['rest_set'][train_i][train_idx]
            X_valid = ids[index]['rest_set'][train_i][test_idx]
            train_index = np.concatenate([ids[index]['rep_set'], X_train])
            test_index = ids[index]['rest_set'][test_i]
            assert len(set(train_index) & set(test_index)) == 0
            # assert len(train_index) + len(test_index) + len(X_valid) == X.shape[0]
            split = {"train": torch.from_numpy(train_index), "valid": (torch.from_numpy(
                X_valid)).type(torch.int64), "test": (torch.from_numpy(test_index)).type(torch.int64)}
            list_split_idx_3.append(split)
        dict_Split_idx[index] = list_split_idx_3.copy()
        index += 1

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_index)
    data.val_mask = get_mask(X_valid)
    data.test_mask = get_mask(test_index)

    return data

    # return dict_Split_idx

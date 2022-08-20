from data import get_dataset, set_fixed_train_val_test_split, set_ratio_train_valid_test_split
from utils import ROOT_DIR
import torch
from torch import Tensor
import numpy as np


def mask_to_index(mask: Tensor) -> Tensor:
    r"""Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.
    """
    return mask.nonzero(as_tuple=False).view(-1)


def get_accuracy(opt):

    dataset = get_dataset(opt, f'{ROOT_DIR}/data', opt['not_lcc'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0]

    if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
        dataset.data = set_fixed_train_val_test_split(np.random.randint(
            0, 1000), dataset.data, num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

    data = dataset.data.to(device)
    split_idx = {"train": mask_to_index(data.train_mask), "valid": mask_to_index(
        data.val_mask), "test": mask_to_index(data.test_mask)}
    train_idx = mask_to_index(data.train_mask).to(device)

    N = data.num_nodes
    labels = data.y.squeeze()
    print(f'Labels: {labels.shape[0]}')
    print(f'Train: {split_idx["train"].shape[0]}')
    print(f'Valid: {split_idx["valid"].shape[0]}')
    print(f'Test: {split_idx["test"].shape[0]}')

    model_outputs = []
    for i in range(opt['runs']):
        model_outputs.append(torch.load(
            f'{ROOT_DIR}/output/{opt["dataset"]}/{i}.pt', map_location='cpu'))

    # Standard Accuracy
    # All Nodes
    std_acc_all_nodes = []
    for i in range(opt['runs']):
        A = np.argmax(model_outputs[i], axis=1)
        std_acc_all_nodes.append((A == labels).sum() / N)
    # Test Nodes
    std_acc_test_nodes = []
    for i in range(opt['runs']):
        A = np.argmax(model_outputs[i][-split_idx["test"].shape[0]:], axis=1)
        std_acc_test_nodes.append(
            (A == labels[-split_idx["test"].shape[0]:]).sum() / A.shape[0])

    # Ensemble Accuracy
    # All Nodes
    L = np.zeros((N, 40))
    for i in range(opt['runs']):
        L = np.add(L, np.log(model_outputs[i]))
    A = np.argmax(L, axis=1)
    ens_acc_all_nodes = (A == labels).sum() / N
    # Test Nodes
    L = np.zeros((len(split_idx['test']), 40))
    for i in range(opt['runs']):
        L = np.add(L, np.log(model_outputs[i][-split_idx["test"].shape[0]:]))
    A = np.argmax(L, axis=1)
    ens_acc_test_nodes = (
        A == labels[-split_idx["test"].shape[0]:]).sum() / A.shape[0]

    # k-in-j accuracy (Hard and Soft Accuracy)
    counts = np.zeros([1, N])
    for i in range(opt['runs']):
        match = (np.argmax(model_outputs[i], axis=1) == labels)
        match = match.detach().numpy()
        match = match.astype(int)
        counts = counts + match
    # soft accuracy
    soft_acc = np.sum((counts > 0).astype(int))/N
    # hard accuracy
    hard_acc = np.sum((counts > 9).astype(int))/N

    # Accuracy Gap
    acc_gap = (np.sum((counts > 9).astype(int))/N) / \
        (np.sum((counts > 0).astype(int))/N)

    # per model accuracy
    matching_models = []
    for i in range(0, opt['runs']+1):
        if i == 0:
            unmatched_model = ((counts[0] == 0).sum()/N) * 100
        else:
            matching_models.append((counts == i).sum() / N)

    result = {"Dataset": opt['dataset'], "Standard Accuracy": {"All Nodes": std_acc_all_nodes, "Test Nodes": std_acc_test_nodes}, "Ensemble Accuracy": {"All Nodes": ens_acc_all_nodes,
                                                                                                                                                        "Test Nodes": ens_acc_test_nodes}, "Soft Accuracy": soft_acc, "Hard Accuracy": hard_acc, "Accuracy Gap": acc_gap, "Unmatched Node Percentage": unmatched_model, "# of matching models": matching_models}
    return result

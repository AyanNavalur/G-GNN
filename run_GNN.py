import json
from utils import ROOT_DIR
from best_params import best_params_dict
from data import get_dataset, set_fixed_train_val_test_split, set_ratio_train_valid_test_split
from logger import Logger
from gnn import GCN, SAGE
from ogb.nodeproppred import Evaluator
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv
import torch
import numpy as np
from torch import Tensor
import os
import time
import argparse
import random
import pathlib
from accuracy import get_accuracy

from torch_geometric.utils import to_scipy_sparse_matrix, index_to_mask  # noqa

# from CGNN import CGNN, get_sym_adj
# from CGNN import train as train_cgnn


def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def add_labels(feat, labels, idx, num_classes, device):
    onehot = torch.zeros([feat.shape[0], num_classes]).to(device)
    if idx.dtype == torch.bool:
        idx = torch.where(idx)[0]  # convert mask to linear index
    onehot[idx, labels.squeeze()[idx]] = 1

    return torch.cat([feat, onehot], dim=-1)


def mask_to_index(mask: Tensor) -> Tensor:
    r"""Converts a mask to an index representation.

    Args:
        mask (Tensor): The mask.
    """
    return mask.nonzero(as_tuple=False).view(-1)


def get_label_masks(data, mask_rate=0.5):
    """
    when using labels as features need to split training nodes into training and prediction
    """
    if data.train_mask.dtype == torch.bool:
        idx = torch.where(data.train_mask)[0]
    else:
        idx = data.train_mask
    mask = torch.rand(idx.shape) < mask_rate
    train_label_idx = idx[mask]
    train_pred_idx = idx[~mask]
    return train_label_idx, train_pred_idx


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return (train_acc, valid_acc, test_acc), out


def print_model_params(model):
    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param.data.shape)


def merge_cmd_args(cmd_opt, opt):
    if cmd_opt['beltrami']:
        opt['beltrami'] = True
    if cmd_opt['function'] is not None:
        opt['function'] = cmd_opt['function']
    if cmd_opt['block'] is not None:
        opt['block'] = cmd_opt['block']
    if cmd_opt['attention_type'] != 'scaled_dot':
        opt['attention_type'] = cmd_opt['attention_type']
    if cmd_opt['self_loop_weight'] is not None:
        opt['self_loop_weight'] = cmd_opt['self_loop_weight']
    if cmd_opt['method'] is not None:
        opt['method'] = cmd_opt['method']
    if cmd_opt['step_size'] != 1:
        opt['step_size'] = cmd_opt['step_size']
    if cmd_opt['time'] != 1:
        opt['time'] = cmd_opt['time']
    if cmd_opt['epoch'] != 100:
        opt['epoch'] = cmd_opt['epoch']
    if not cmd_opt['not_lcc']:
        opt['not_lcc'] = False
    if cmd_opt['num_splits'] != 1:
        opt['num_splits'] = cmd_opt['num_splits']


def set_seed(_seed):
    torch.manual_seed(_seed)
    torch.cuda.manual_seed_all(_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(_seed)
    random.seed(_seed)
    os.environ['PYTHONHASHSEED'] = str(_seed)


def main(cmd_opt):
    # try:
    #   # best_opt = best_params_dict[cmd_opt['dataset']]
    #   # opt = {**cmd_opt, **best_opt}
    #   merge_cmd_args(cmd_opt, opt)
    # except KeyError:
    #   opt = cmd_opt
    opt = cmd_opt

    dataset = get_dataset(opt, f'{ROOT_DIR}/data', opt['not_lcc'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = dataset[0]

    if args.use_sage:
        model = SAGE(data.num_features, opt['hidden_channels'],
                     dataset.num_classes, opt['num_layers'],
                     opt['dropout']).to(device)
    else:
        model = GCN(data.num_features, opt['hidden_channels'],
                    dataset.num_classes, opt['num_layers'],
                    opt['dropout']).to(device)

    if not opt['planetoid_split'] and opt['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
        dataset.data = set_fixed_train_val_test_split(np.random.randint(
            0, 1000), dataset.data, num_development=5000 if opt["dataset"] == "CoauthorCS" else 1500)

    data = dataset.data.to(device)
    split_idx = {"train": mask_to_index(data.train_mask), "valid": mask_to_index(
        data.val_mask), "test": mask_to_index(data.test_mask)}
    train_idx = mask_to_index(data.train_mask).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(opt['num_splits'] * opt['runs'], args)
    # internal_logger = Logger(10, args)
    logger_report = Logger(opt['num_splits'], args)
    logger_test = Logger(opt['num_splits'] * opt['runs'], args)

    parameters = [p for p in model.parameters() if p.requires_grad]
    print(sum(p.numel() for p in model.parameters()))
    print_model_params(model)
    optimizer = get_optimizer(
        opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    best_time = best_epoch = train_acc = val_acc = test_acc = 0

    random_state = np.random.RandomState(5555)
    run_seeds_lst = random_state.randint(0, 1000000, opt['runs'])

    # create directories if do not exist
    pathlib.Path(
        f'{ROOT_DIR}/output/{opt["dataset"]}').mkdir(parents=True, exist_ok=True)

    for run in range(args.runs):
        set_seed(run_seeds_lst[run])
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_valid = 0
        best_run = 0
        best_out = None
        best_model = None
        losses = []
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result, out = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)
            train_acc, valid_acc, test_acc = result
            if valid_acc > best_valid:
                best_valid = valid_acc
                best_out = out.cpu().exp()
                best_run = epoch

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        # to save the output of model
        torch.save(best_out, f'{ROOT_DIR}/output/{opt["dataset"]}/{run}.pt')
        logger.print_statistics(run)
    train_final, valid_final, test_final = logger.print_statistics()

    accuracy_result = get_accuracy(opt)
    accuracy_file = open(args.accuracypath, "a+", encoding='utf-8')
    accuracy_file.write(json.dumps(accuracy_result) + '\n')
    accuracy_file.close()

    result = {"Dataset": args.external_dataset, "Model": args.booster, "train_ratio": int(args.train_ratio),
              "Average Train": train_final, " Average Valid": valid_final, " Average Test": test_final}
    result_file = open(args.outputpath, "a+", encoding='utf-8')
    result_file.write(json.dumps(result) + '\n')
    result_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cora_defaults', action='store_true',
                        help='Whether to run with best params for cora. Overrides the choice of dataset')
    # data args
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv',
                        help='Cora, Citeseer, Pubmed, Computers, Photo, CoauthorCS, ogbn-arxiv, WikiCS')
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float,
                        default=1.0, help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels',
                        action='store_true', help='Also diffuse labels')
    parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                        help='use the 10 fixed splits from '
                             'https://arxiv.org/abs/2002.05287')
    parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                        help='the number of splits to repeat the results on')
    parser.add_argument('--label_rate', type=float, default=0.5,
                        help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true',
                        help='use planetoid splits for Cora/Citeseer/Pubmed')
    parser.add_argument('--not_lcc', action='store_false',
                        help='use largest coonected component')
    parser.add_argument("--global_random_seed", type=int, default=2021,
                        help="Random seed (for reproducibility).")
    parser.add_argument("--outputpath", type=str, default="empty.json",
                        help="outputh file path to save the result")
    parser.add_argument("--accuracypath", type=str, default="accuracy.json",
                        help="accuracy file path to save the accuracy result")
    parser.add_argument("--train_ratio", type=float, default=1.,
                        help="the start value of the train ratio (inclusive).")
    parser.add_argument("--split_policy", type=str, default='fixed',
                        help="")
    # GNN args
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    # parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    # parser.add_argument("--splits", type=int, default=5,
    #                     help="The number of re-shuffling & splitting for each train ratio.")

    # parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    # parser.add_argument('--fc_out', dest='fc_out', action='store_true',
    #                     help='Add a fully connected layer to the decoder.')
    # parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    # parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    # parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='One from sgd, rmsprop, adam, adagrad, adamax.')
    # parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4,
                        help='Weight decay for optimization')
    # parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
    # parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    # parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    # parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
    #                     help='apply sigmoid before multiplying by alpha')
    # parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    # parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    # parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT')
    # parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
    #                     help='Add a fully connected layer to the encoder.')
    # parser.add_argument('--add_source', dest='add_source', action='store_true',
    #                     help='If try get rid of alpha param and the beta*x0 source term')
    # parser.add_argument('--cgnn', dest='cgnn', action='store_true', help='Run the baseline CGNN model from ICML20')

    args = parser.parse_args()

    opt = vars(args)

    main(opt)

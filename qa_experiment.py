"""
This script computes results for the scene graph experiments using the Neural Tree (message passing on H-trees) or the
vanilla architectures (message passing on original graphs).
The dataset split is generated randomly for each run with fixed random seed.
"""
from neural_tree.utils.base_training_job import BaseTrainingJob, print_log
from statistics import mean, stdev
from os import mkdir, path
import random
from datetime import datetime
import torch
import numpy as np
import argparse
from neural_tree.utils import data_utils
from neural_tree.utils import utils
from neural_tree.utils import parser_utils
import os
DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}

def get_devices(use_cuda):
    """Get the devices to put the data and the model based on whether to use GPUs and, if so, how many of them are available."""
    if torch.cuda.device_count() >= 2 and use_cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
        print("device0: {}, device1: {}".format(device0, device1))
    elif torch.cuda.device_count() == 1 and use_cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    return device0, device1

def load_data(args, devices, kg):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    #########################################################
    # Construct the dataset
    #########################################################
    dataset = data_utils.DataLoader(args.train_statements, args.train_adj,
        args.dev_statements, args.dev_adj,
        args.test_statements, args.test_adj,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        device=devices,
        model_name=args.encoder,
        max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
        is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
        subsample=args.subsample, n_train=args.n_train, debug=args.debug, cxt_node_connects_all=args.cxt_node_connects_all, kg=kg)

    return dataset.train(), dataset.dev(), dataset.test()

def main(args):
    ############## dataset ############################################
    project_dir = path.dirname(path.abspath(__file__))

    devices = get_devices(args.cuda)
    kg = "cpnet"
    if args.dataset == "medqa_usmle":
        kg = "ddb"
        
    csqa_file = "data/csqa.pt"
    if os.path.isfile(csqa_file):
        dataset = torch.load(csqa_file)
    else:
        dataset = load_data(args, devices, kg)
        torch.save(dataset, csqa_file)
    task = 'graph'
    
    ############## run control #########################################
    num_runs = 100
    early_stop_window = -1  # setting to -1 will disable early stop
    verbose = False
    log_folder_master = project_dir + '/log'
    
    ############### algorithm ##########################################
    algorithm = 'neural_tree'
    # algorithm = 'original'
    
    ############### parameters #########################################
    if task == "node":
        train_node_ratio = 0.7
        val_node_ratio = 0.1
        test_node_ratio = 0.2
    network_params = {'conv_block': 'GraphSAGE',
                      'hidden_dim': 128,
                      'num_layers': 1,
                      'GAT_hidden_dims': [128, 128],
                      'GAT_heads': [6, 1],
                      'GAT_concats': [True, False],
                      'dropout': 0.25}
    optimization_params = {'lr': 0.001,
                           'num_epochs': 1000,
                           'weight_decay': 0.001}
    dataset_params = {'batch_size': 1,
                      'shuffle': False}
    neural_tree_params = {'min_diameter': 1,      # diameter=0 means the H-tree is disconnected
                          'max_diameter': None,
                          'sub_graph_radius': None}
    
    #####################################################################


    random.seed(0)

    # setup log folder, parameter and accuracy files
    log_folder = log_folder_master + datetime.now().strftime('/%Y%m%d-%H%M%S')
    mkdir(log_folder)
    print('Starting graph classification on CommonsenseQA, OpenbookQA, and MedQA datasets using {}. Results saved to {}'.
          format(algorithm, log_folder))
    f_param = open(log_folder + '/parameter.txt', 'w')
    f_log = open(log_folder + '/accuracy.txt', 'w')
    
    if task == "node":
        print('train_node_ratio: {}'.format(train_node_ratio), file=f_param)
        print('val_node_ratio: {}'.format(val_node_ratio), file=f_param)
        print('test_node_ratio: {}'.format(test_node_ratio), file=f_param)

    # run experiment
    test_accuracy_list = []
    val_accuracy_list = []
    for i in range(num_runs):
        print("run number: ", i)
        
        if task == "node":
            # data split
            dataset.generate_node_split(train_node_ratio=train_node_ratio, val_node_ratio=val_node_ratio,
                                    test_node_ratio=test_node_ratio)

        # training
        train_job = BaseTrainingJob(algorithm, task, dataset, network_params, neural_tree_params)
        model, best_acc = train_job.train(log_folder + '/' + str(i), optimization_params, dataset_params,
                                          early_stop_window=early_stop_window, verbose=verbose)

        if i == 0:
            # save parameters in train_job to parameter.txt (only need to save once)
            train_job.print_training_params(f=f_param)
            f_param.close()

        if isinstance(best_acc, tuple):
            val_accuracy_list.append(best_acc[0] * 100)
            test_accuracy_list.append(best_acc[1] * 100)
        else:
            raise RuntimeError('Validation set is empty.')

    print_log('Validation accuracy: {}'.format(val_accuracy_list), f_log)
    print_log('Test accuracy: {}'.format(test_accuracy_list), f_log)
    print_log("Average test accuracy from best validation accuracy ({:.2f} +/- {:.2f} %) over {} runs is "
              "{:.2f} +/- {:.2f} %. ".format(mean(val_accuracy_list), stdev(val_accuracy_list), num_runs,
                                             mean(test_accuracy_list), stdev(test_accuracy_list)), f_log)
    max_val_accuracy = max(val_accuracy_list)
    print_log('Best validation accuracy over all runs: {:.2f} %, corresponding test accuracy: {:.2f} %'.
              format(max_val_accuracy, test_accuracy_list[val_accuracy_list.index(max_val_accuracy)]), file=f_log)
    f_log.close()
    print('End of scene_graph_experiment.py. Results saved to: {}\n'.format(log_folder))


if __name__ == '__main__':
    __spec__ = None

    parser = parser_utils.get_parser()
    args, _ = parser.parse_known_args()

    # General
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/greaselm/', help='model output directory')
    parser.add_argument('--save_model', default=True, type=utils.bool_flag, help="Whether to save model checkpoints or not.")
    parser.add_argument('--load_model_path', default=None, help="The model checkpoint to load in the evaluation mode.")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument("--run_name", required=True, type=str, help="The name of this experiment run.")
    parser.add_argument("--resume_checkpoint", default=None, type=str,
                        help="The checkpoint to resume training from.")
    parser.add_argument('--use_wandb', default=False, type=utils.bool_flag, help="Whether to use wandb or not.")
    parser.add_argument("--resume_id", default=None, type=str, help="The wandb run id to resume if `resume_checkpoint` is not None or 'None'.")

    # Data
    parser.add_argument('--train_adj', default=f'{args.data_dir}/{args.dataset}/graph/train.graph.adj.pk', help="The path to the retrieved KG subgraphs of the training set.")
    parser.add_argument('--dev_adj', default=f'{args.data_dir}/{args.dataset}/graph/dev.graph.adj.pk', help="The path to the retrieved KG subgraphs of the dev set.")
    parser.add_argument('--test_adj', default=f'{args.data_dir}/{args.dataset}/graph/test.graph.adj.pk', help="The path to the retrieved KG subgraphs of the test set.")
    parser.add_argument('--max_node_num', default=200, type=int, help="Max number of nodes / the threshold used to prune nodes.")
    parser.add_argument('--subsample', default=1.0, type=float, help="The ratio to subsample the training set.")
    parser.add_argument('--n_train', default=-1, type=int, help="Number of training examples to use. Setting it to -1 means using the `subsample` argument to determine the training set size instead; otherwise it will override the `subsample` argument.")

    # Model architecture
    parser.add_argument('-k', '--k', default=5, type=int, help='The number of GreaseLM layers')
    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads of the final graph nodes\' pooling')
    parser.add_argument('--gnn_dim', default=256, type=int, help='dimension of the GNN layers')
    parser.add_argument('--fc_dim', default=200, type=int, help='number of FC hidden units (except for the MInt operators)')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of hidden layers of the final MLP')
    parser.add_argument('--freeze_ent_emb', default=True, type=utils.bool_flag, nargs='?', const=True, help='Whether to freeze the entity embedding layer.')
    parser.add_argument('--ie_dim', default=200, type=int, help='number of the hidden units of the MInt operator.')
    parser.add_argument('--info_exchange', default=True, choices=[True, False, "every-other-layer"], type=utils.bool_str_flag, help="Whether we have the MInt operator in every GreaseLM layer or every other GreaseLM layer or not at all.")
    parser.add_argument('--ie_layer_num', default=1, type=int, help='number of hidden layers in the MInt operator')
    parser.add_argument("--sep_ie_layers", default=False, type=utils.bool_flag, help="Whether to share parameters across the MInt ops across differernt GreaseLM layers or not. Setting it to `False` means sharing.")
    parser.add_argument('--random_ent_emb', default=False, type=utils.bool_flag, nargs='?', const=True, help='Whether to use randomly initialized learnable entity embeddings or not.')
    parser.add_argument("--cxt_node_connects_all", default=False, type=utils.bool_flag, help="Whether to connect the interaction node to all the retrieved KG nodes or only the linked nodes.")

    parser.add_argument('--n_ntype', default=4, type=int, help='number of node types')
    parser.add_argument('--n_etype', default=38, type=int, help='number of edge types')

    # Regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # Optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='Learning rate of parameters not in LM')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=2, type=int)
    parser.add_argument('--unfreeze_epoch', default=4, type=int, help="Number of the first few epochs in which LM’s parameters are kept frozen.")
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')

    # Additional Model Arguments
    parser.add_argument("--model_name_or_path", default=f"{args.encoder}", type=str, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--config_name", default=None, type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str, help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--use_fast_tokenizer", default=True, type=bool, help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.")
    parser.add_argument("--model_revision", default="main", type=str, help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--use_auth_token", default=False, type=bool, help="Will use the token generated when running `transformers-cli login` (necessary to use this script with private models).")
    parser.add_argument("--prefix", default=False, type=bool, help="Will use P-tuning v2 during training")
    parser.add_argument("--prompt", default=False, type=bool, help="Will use prompt tuning during training")
    parser.add_argument("--pre_seq_len", default=128, type=int, help="The length of prompt")
    parser.add_argument("--prefix_projection", default=False, type=bool, help="Apply a two-layer MLP head over the prefix embeddings")
    parser.add_argument("--prefix_hidden_size", default=256, type=int, help="The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="The dropout probability used in the models")

    args = parser.parse_args()
    main(args)
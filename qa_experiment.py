"""
This script computes results for the scene graph experiments using the Neural Tree (message passing on H-trees) or the
vanilla architectures (message passing on original graphs).
The dataset split is generated randomly for each run with fixed random seed.
"""
from neural_tree.utils.base_job import BaseJob, print_log
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
    'csqa': 1e-5,
    'obqa': 1e-5,
    'medqa_usmle': 1e-5,
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
        device=devices,
        model_name=args.encoder,
        max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
        is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
        subsample=args.subsample, n_train=args.n_train, debug=args.debug, cxt_node_connects_all=args.cxt_node_connects_all, kg=kg)

    return dataset.train(), dataset.dev(), dataset.test()

def load_params(args):
    
    network_params = {'aggr_encoder':args.encoder,
                      'conv_block': 'GraphSAGE',
                      'dropout': 0.25, 
                      'pre_seq_len':args.pre_seq_len, 
                      'prefix_tuning': args.prefix_tuning, 
                      'prefix_projection': args.prefix_projection, 
                      'prefix_hidden_size': args.prefix_hidden_size,
                      'hidden_layer_retention_rate': args.hidden_layer_retention_rate,
                      #'visualize': args.mode == 'eval',
                      'visualize': False,
                      'graph_pooling': args.graph_pooling}
    optimization_params = {'lr': args.learning_rate,
                           'num_epochs': args.n_epochs,
                           'weight_decay': 0.001, 
                           'lr_decay_epochs': args.lr_decay_epochs,
                           'lr_decay_rate': args.lr_decay_rate,
                           'refreeze_epochs':args.refreeze_epochs, 
                           'resume_checkpoint': args.resume_checkpoint,
                           'save_model': args.save_model,
                           'save_dir': args.save_dir
                           }
    dataset_params = {'mini_batch_size': args.mini_batch_size,
                      'batch_size': args.batch_size,
                      'shuffle': True,
                      #'num_choices': args.num_choices,
                      'seq_len':args.max_seq_len,
                      'max_node_num':args.max_node_num, 
                      'tree_width':args.tree_width,
                      'dataset': args.dataset,
                      'inhouse': args.inhouse}
    neural_tree_params = {'min_diameter': 1,      # diameter=0 means the H-tree is disconnected
                          'max_diameter': None,
                          'sub_graph_radius': None}

    return network_params, optimization_params, dataset_params, neural_tree_params

def load_cache(args):
    devices = get_devices(args.cuda)
    kg = "cpnet"
    if args.dataset == "medqa_usmle":
        kg = "ddb"
        
    graph_file = "{}_graph_{}_n{}_sl{}_tw{}{}.pt".format(args.dataset, args.encoder, args.max_node_num, args.max_seq_len, args.tree_width, '_inhouse' if args.inhouse else '')
    graph_path = "data/{}".format(graph_file)
    if os.path.isfile('{}.zip'.format(graph_path)):
        os.system('unzip {}.zip -d data'.format(graph_path))
        dataset = torch.load(graph_path)
        os.system('rm {}'.format(graph_path))
    else:
        dataset = load_data(args, devices, kg)
        torch.save(dataset, graph_path)
        os.system('cd data && zip -m {}.zip {} '.format(graph_file, graph_file))
        #os.system('rm {}'.format(graph_path))
    return dataset

def train(args):
    ############## dataset ############################################
    project_dir = path.dirname(path.abspath(__file__))

    
        
    dataset = load_cache(args)
    
    task = 'graph'
    
    ############## run control #########################################
    num_runs = 100
    early_stop_window = -1  # setting to -1 will disable early stop
    verbose = False
    log_folder_master = project_dir + '/log'
    save_folder_master = project_dir + '/run'
    ############### algorithm ##########################################
    algorithm = 'neural_tree'
    # algorithm = 'original'
    
    ############### parameters #########################################
    if task == "node":
        train_node_ratio = 0.7
        val_node_ratio = 0.1
        test_node_ratio = 0.2

    #####################################################################


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    #random.seed(0)

    # setup log folder, parameter and accuracy files
    curr_dt = datetime.now()
    log_folder = log_folder_master + curr_dt.strftime('/%Y%m%d-%H%M%S') + '_' + args.dataset + '_' + args.encoder
    os.system('mkdir -p ' + log_folder)
    
    if args.save_dir is None:
        args.save_dir = save_folder_master + '/' + args.dataset + '/' + curr_dt.strftime('/%Y%m%d-%H%M%S') + '_' + args.dataset + '_' + args.encoder
        os.system('mkdir -p ' + args.save_dir)
    
    
    print('Starting graph classification on CommonsenseQA, OpenbookQA, and MedQA datasets using {}. Results saved to {}'.
          format(algorithm, log_folder))
    f_param = open(log_folder + '/parameter.txt', 'w')
    f_log = open(log_folder + '/accuracy.txt', 'w')
    
    if task == "node":
        print('train_node_ratio: {}'.format(train_node_ratio), file=f_param)
        print('val_node_ratio: {}'.format(val_node_ratio), file=f_param)
        print('test_node_ratio: {}'.format(test_node_ratio), file=f_param)


    network_params, optimization_params, dataset_params, neural_tree_params = load_params(args)

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
        train_job = BaseJob(algorithm, task, dataset, network_params, neural_tree_params, dataset_params, optimization_params)
        model, best_acc = train_job.train(log_folder + '/' + str(i), early_stop_window=early_stop_window, verbose=verbose)

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

def evaluate(args):
    task = 'graph'
    algorithm = 'neural_tree'
    dataset = load_cache(args)
    network_params, optimization_params, dataset_params, neural_tree_params = load_params(args)
    eval_job = BaseJob(algorithm, task, dataset, network_params, neural_tree_params, dataset_params, optimization_params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = eval_job.get_dataloader()
    checkpoint = torch.load(args.load_model_path, map_location=device)
    epoch = checkpoint['epoch']
    global_step = checkpoint['global_step']
    net = eval_job.get_net()
    net.to(device)
    net.load_state_dict(checkpoint["model"], strict=False)
    test_result = eval_job.test(test_loader)
    print('test result:', test_result, 'epoch:', epoch, 'global_step:', global_step)
    

if __name__ == '__main__':
    __spec__ = None

    parser = parser_utils.get_parser()
    args, _ = parser.parse_known_args()

    # General
    parser.add_argument('--mode', default='train', choices=['train', 'eval'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=None, help='model output directory')
    parser.add_argument('--save_model', default=True, type=utils.bool_flag, help="Whether to save model checkpoints or not.")
    parser.add_argument('--load_model_path', default=None, help="The model checkpoint to load in the evaluation mode.")
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument("--resume_checkpoint", default=None, type=str,
                        help="The checkpoint to resume training from.")
    parser.add_argument('--use_wandb', default=False, type=utils.bool_flag, help="Whether to use wandb or not.")
    parser.add_argument("--resume_id", default=None, type=str, help="The wandb run id to resume if `resume_checkpoint` is not None or 'None'.")

    # Data
    parser.add_argument('--train_adj', default=f'{args.data_dir}/{args.dataset}/graph/train.graph.adj.pk', help="The path to the retrieved KG subgraphs of the training set.")
    parser.add_argument('--dev_adj', default=f'{args.data_dir}/{args.dataset}/graph/dev.graph.adj.pk', help="The path to the retrieved KG subgraphs of the dev set.")
    parser.add_argument('--test_adj', default=f'{args.data_dir}/{args.dataset}/graph/test.graph.adj.pk', help="The path to the retrieved KG subgraphs of the test set.")
    parser.add_argument('--max_node_num', default=4, type=int, help="Max number of nodes / the threshold used to prune nodes.")
    parser.add_argument('--subsample', default=1.0, type=float, help="The ratio to subsample the training set.")
    parser.add_argument('--n_train', default=-1, type=int, help="Number of training examples to use. Setting it to -1 means using the `subsample` argument to determine the training set size instead; otherwise it will override the `subsample` argument.")
    #parser.add_argument('--num_choices', default=5, type=int, help="Number of choices")
    parser.add_argument('--tree_width', default=1, type=int, help="Tree width")
    parser.add_argument("--cxt_node_connects_all", default=False, type=utils.bool_flag, help="Whether to connect the interaction node to all the retrieved KG nodes or only the linked nodes.")

    # Model architecture
    parser.add_argument('--freeze_ent_emb', default=True, type=utils.bool_flag, nargs='?', const=True, help='Whether to freeze the entity embedding layer.')
    parser.add_argument('--random_ent_emb', default=False, type=utils.bool_flag, nargs='?', const=True, help='Whether to use randomly initialized learnable entity embeddings or not.')
    
    parser.add_argument('--n_ntype', default=4, type=int, help='number of node types')
    parser.add_argument('--n_etype', default=38, type=int, help='number of edge types')

    # Regularization
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # Optimization
    parser.add_argument('-lr', '--learning_rate', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='Learning rate of parameters not in LM')
    parser.add_argument('-mbs', '--mini_batch_size', default=8, type=int)
    parser.add_argument('--unfreeze_epochs', default=4, type=int, help="Number of the first few epochs in which LM’s parameters are kept frozen.")
    parser.add_argument('--refreeze_epochs', default=100, type=int)
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--lr_decay_epochs', default=100, type=int)
    parser.add_argument('--lr_decay_rate', default=0.1, type=float)

    # Additional Model Arguments
    parser.add_argument("--model_name_or_path", default=f"{args.encoder}", type=str, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--config_name", default=None, type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str, help="Where do you want to store the pretrained models downloaded from huggingface.co")
    parser.add_argument("--use_fast_tokenizer", default=True, type=bool, help="Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.")
    parser.add_argument("--model_revision", default="main", type=str, help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--use_auth_token", default=False, type=bool, help="Will use the token generated when running `transformers-cli login` (necessary to use this script with private models).")
    parser.add_argument("--prefix_tuning", default=False, type=bool, help="Will use P-tuning v2 during training")
    parser.add_argument("--pre_seq_len", default=32, type=int, help="The length of prompt")
    parser.add_argument("--prefix_projection", default=False, type=bool, help="Apply a two-layer MLP head over the prefix embeddings")
    parser.add_argument("--prefix_hidden_size", default=256, type=int, help="The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="The dropout probability used in the models")
    parser.add_argument("-hlrr", "--hidden_layer_retention_rate", default=1.0, type=float, help="The retention ratio of pretrained layers")
    parser.add_argument("--graph_pooling", default="context", type=str, help="Graph pooling (leaf, context, root, or all) at the last layer")


    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
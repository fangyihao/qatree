from model import get_qa_tree_network
import sys
import time
from copy import deepcopy
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tensorboardX import SummaryWriter
import os
from transformers import AutoConfig
from transformers.models.roberta.modeling_roberta import RobertaModel
from torch import nn
from torch.nn import CrossEntropyLoss
from util.util import Dict2Class
from transformers.configuration_utils import PretrainedConfig

from torch_geometric.data import Data
#from torch.distributed.fsdp import (
#   FullyShardedDataParallel,
#   CPUOffload,
#)
#from torch.distributed.fsdp.wrap import (
#   default_auto_wrap_policy,
#)
#from torch.nn.parallel import DistributedDataParallel as DDP
#from patrickstar.runtime import initialize_engine
#from patrickstar.utils import get_rank
class BaseJob:
    def __init__(self, dataset, network_params=None, neural_tree_params=None, dataset_params=None, optimization_params=None):

        # initialize training parameters
        self.__training_params = self.create_default_params()
        self.update_training_params(network_params=network_params, dataset_params=dataset_params, optimization_params=optimization_params, neural_tree_params=neural_tree_params)

        #self.remove_unused_network_params(self.__training_params['network_params'])

        self.__dataset = dataset
        
        # initialize network
        self.__net = self.initialize_network()

    @staticmethod
    def create_default_params():

        network_params = {}
        optimization_params = {}
        dataset_params = {}
        neural_tree_params = {}

        training_params = {'network_params': network_params, 'optimization_params': optimization_params,
                           'dataset_params': dataset_params, 'neural_tree_params': neural_tree_params}
        return training_params

    @staticmethod
    def remove_unused_network_params(network_params):
        if network_params['conv_block'] != 'GAT':
            removed_params = ['GAT_hidden_dims', 'GAT_heads', 'GAT_concats']
        else:
            removed_params = ['hidden_dim', 'num_layers']
        for param in removed_params:
            network_params.pop(param)
        return network_params

    def print_training_params(self, f=sys.stdout):
        for params, params_dict in self.__training_params.items():
            print(params, file=f)
            for param_name, value in params_dict.items():
                print('   {}: {}'.format(param_name, value), file=f)

    def update_training_params(self, network_params=None, optimization_params=None, dataset_params=None,
                               neural_tree_params=None):
        if network_params is not None:
            for key in network_params:
                self.__training_params['network_params'][key] = network_params[key]
        if optimization_params is not None:
            for key in optimization_params:
                self.__training_params['optimization_params'][key] = optimization_params[key]
        if dataset_params is not None:
            for key in dataset_params:
                self.__training_params['dataset_params'][key] = dataset_params[key]
        if neural_tree_params is not None:
            for key in neural_tree_params:
                self.__training_params['neural_tree_params'][key] = neural_tree_params[key]

    def initialize_network(self):
    
        #model_name_or_path = "roberta-large"
        #model_name_or_path = "roberta-base"
        model_name_or_path = self.__training_params['network_params']['aggr_encoder']
        #model_name_or_path = "bert-base-uncased"
        
        if model_name_or_path.startswith("glm"):
            config = PretrainedConfig(**{"name_or_path": "glm-large", 
                                 "vocab_size": 50304, 
                                 "hidden_size": 1024, 
                                 "hidden_dropout_prob": 0.1, 
                                 "num_hidden_layers": 24, 
                                 "num_attention_heads": 16,
                                 "max_memory_length": 0,
                                 "attention_dropout_prob": 0.1, 
                                 "output_dropout_prob": 0.1,
                                 "attention_scale": 1.0,
                                 "relative_encoding": True,
                                 "block_position_encoding": False,
                                 "checkpoint_activations": True,
                                 "checkpoint_num_layers": 1
                                 })
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path,#"bert-base-uncased", #args.model_name_or_path
                revision="main"
                )
        
        config.prefix_tuning = self.__training_params['network_params']['prefix_tuning']
        config.prefix_projection = self.__training_params['network_params']['prefix_projection']
        config.prefix_hidden_size = self.__training_params['network_params']['prefix_hidden_size']
        config.seq_len = self.__training_params['dataset_params']['seq_len']
        config.pre_seq_len = self.__training_params['network_params']['pre_seq_len']
        #config.num_choices = self.__training_params['dataset_params']['num_choices']
        config.hidden_layer_retention_rate = self.__training_params['network_params']['hidden_layer_retention_rate']
        config.first_attn_mask_layers = int(config.num_hidden_layers * config.hidden_layer_retention_rate)
        config.graph_pooling = self.__training_params['network_params']['graph_pooling'] # context
        config.visualize = self.__training_params['network_params']['visualize']
        config.aggr = "cat" # "cat" or "mean"
        config.gradient_checkpointing = self.__training_params['optimization_params']['gradient_checkpointing']
        config.random_walk = self.__training_params['network_params']['random_walk']
        config.random_walk_sample_rate = self.__training_params['network_params']['random_walk_sample_rate']
        config.contrastive_loss = self.__training_params['network_params']['contrastive_loss']
        config.contrastive_loss_scalar = self.__training_params['network_params']['contrastive_loss_scalar']
        config.isotropy_loss_scalar = self.__training_params['network_params']['isotropy_loss_scalar']
        config.cross_entropy_loss_scalar = self.__training_params['network_params']['cross_entropy_loss_scalar']
        print(config)
        
        cls = get_qa_tree_network(config)
        
        return cls.from_pretrained(
            model_name_or_path,#"bert-base-uncased", 
            config = config,
            #conv_block = self.__training_params['network_params']['conv_block']
        )
        

    def get_dataset(self):
        return self.__dataset
    
    def get_net(self):
        return self.__net
    
    def count_parameters(self, model):
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_fixed_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
        print('num_trainable_params:', num_trainable_params)
        print('num_fixed_params:', num_fixed_params)
    
        print('num_total_params:', num_trainable_params + num_fixed_params)


    def get_dataloader(self):
        
        # create data loader
        train_loader = DataLoader(self.__dataset[0],
                                  batch_size=self.__training_params['dataset_params']['mini_batch_size'],
                                  shuffle=self.__training_params['dataset_params']['shuffle'])
        val_loader = DataLoader(self.__dataset[1],
                                batch_size=self.__training_params['dataset_params']['mini_batch_size'],
                                shuffle=self.__training_params['dataset_params']['shuffle'])
        test_loader = DataLoader(self.__dataset[2],
                                 batch_size=self.__training_params['dataset_params']['mini_batch_size'],
                                 shuffle=self.__training_params['dataset_params']['shuffle'])
    
        return train_loader, val_loader, test_loader

    def train(self, log_folder, early_stop_window=-1, verbose=False):
        # update parameters
        #self.update_training_params(optimization_params=optimization_params, dataset_params=dataset_params)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        '''
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        world_size = n_gpus
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        # initialize the process group
        import torch.distributed as dist
        dist.init_process_group("gloo", rank=0, world_size=world_size)
        self.__net = DDP(self.__net)
        self.__net = FullyShardedDataParallel(
           self.__net,
           fsdp_auto_wrap_policy=default_auto_wrap_policy,
           cpu_offload=CPUOffload(offload_params=True),
        )
        '''
        
        train_loader, val_loader, test_loader = self.get_dataloader()
        
        
        print("lr:",self.__training_params['optimization_params']['lr'])
        opt = optim.Adam(self.__net.parameters(), lr=self.__training_params['optimization_params']['lr'],
                         weight_decay=self.__training_params['optimization_params']['weight_decay'])

        self.__net.to(device)

        # resume checkpoint
        resume = self.__training_params['optimization_params']['resume_checkpoint'] is not None \
            and self.__training_params['optimization_params']['resume_checkpoint'] != "None"
        if resume:
            print("loading from checkpoint: {}".format(self.__training_params['optimization_params']['resume_checkpoint']))
            self.__training_params['optimization_params']['save_dir'] = os.path.dirname(self.__training_params['optimization_params']['resume_checkpoint'])

            checkpoint = torch.load(self.__training_params['optimization_params']['resume_checkpoint'], map_location=device)
            last_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            self.__net.load_state_dict(checkpoint["model"], strict=False)
            opt.load_state_dict(checkpoint["optimizer"])
            # workaround for Pytorch 1.12.0
            opt.param_groups[0]['capturable'] = True

        else:
            last_epoch = -1
            global_step = 0
            
        
        lr_decay_epochs=self.__training_params['optimization_params']['lr_decay_epochs']
        lr_decay_rate=self.__training_params['optimization_params']['lr_decay_rate']

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=lr_decay_epochs, gamma=lr_decay_rate, last_epoch=last_epoch)

        if resume:
            scheduler.load_state_dict(checkpoint["scheduler"])
        
        # move training to gpu if available
        
        #self.__net = nn.DataParallel(self.__net)

        '''
        config = {
            # The same format as optimizer config of DeepSpeed
            # https://www.deepspeed.ai/docs/config-json/#optimizer-parameters
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 5e-5,
                    "betas": (0.9, 0.999),
                    "eps": 1e-6,
                    "weight_decay": 0,
                    "use_hybrid_adam": True,
                },
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 2 ** 3,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "default_chunk_size": 64 * 1024 * 1024,
            "release_after_init": False,
            "use_cpu_embedding": False,
        }
        
        self.__net, opt = initialize_engine(model_func=lambda: self.__net, local_rank=get_rank(), config=config)
        '''
        
        #self.count_parameters(self.__net)

        #loss_fct = CrossEntropyLoss(reduction='mean')
        # train
        max_val_acc = 0
        max_test_acc = 0  # If val_loader is not None, compute using the model weights that lead to best_val_acc
        best_model_state = None
        writer = SummaryWriter(log_folder)
        if val_loader is None:
            early_stop_window = -1  # do not use early stopping if there's no validation set

        tic = time.perf_counter()
        early_stop_step = 0
        for epoch in range(last_epoch + 1, self.__training_params['optimization_params']['num_epochs']):

            if epoch == self.__training_params['optimization_params']['refreeze_epochs']:
                for name, param in self.__net.named_parameters():
                    if 'prefix_encoder' not in name:
                        param.requires_grad = False
                        
            self.count_parameters(self.__net)
            
            early_stop_step += 1
            total_loss = 0.
            
            self.__net.train()
            
            num_batches = len(train_loader)
            opt.zero_grad()
            for i, batch in enumerate(train_loader):
                #opt.zero_grad()
                batch = batch.to(device)
                pred, loss = self.__net(batch)
                #label = batch.y
                

                #loss = loss_fct(pred, label)
                loss.backward()

                
                #print("multiplier:", (self.__training_params['dataset_params']['batch_size']//self.__training_params['dataset_params']['mini_batch_size']))
                
                if i % (self.__training_params['dataset_params']['batch_size']//self.__training_params['dataset_params']['mini_batch_size']) == 0 or i == num_batches - 1:
                    opt.step()
                    opt.zero_grad()
                    global_step += 1
                    
                    '''
                    if epoch == 2:
                        val_result = self.test(val_loader, is_validation=True)
                        print('validation result:', val_result, 'epoch:', epoch)
                        self.__net.train()
                    '''
                #if i % 100 == 0:
                #    print('loss:', loss.item(), 'batch:', i)

                total_loss += loss.item() * batch.num_graphs
            total_loss /= len(train_loader.dataset)
            
            print('loss:', total_loss, 'epoch:', epoch)
            writer.add_scalar('loss', total_loss, epoch)

            writer.add_scalar('lr', opt.param_groups[0]["lr"], epoch)
            scheduler.step()

            if verbose:
                train_result = self.test(train_loader, is_train=True)
                writer.add_scalar('train result', train_result, epoch)

            # validation and testing
            if val_loader is not None:
                val_result = self.test(val_loader, is_validation=True)
                print('validation result:', val_result, 'epoch:', epoch)
                writer.add_scalar('validation result', val_result, epoch)
                if val_result > max_val_acc:
                    max_val_acc = val_result
                    best_model_state = deepcopy(self.__net.state_dict())
                    early_stop_step = 0
                if verbose and (epoch + 1) % 10 == 0:
                    print('Epoch {:03}. Loss: {:.4f}. Train accuracy: {:.4f}. Validation accuracy: {:.4f}.'
                          .format(epoch, total_loss, train_result, val_result))
                if early_stop_step == early_stop_window and epoch > early_stop_window:
                    if verbose:
                        print('Early stopping condition reached at {} epoch.'.format(epoch))
                    break
            else:
                test_result = self.test(test_loader)
                print('test result:', test_result, 'epoch:', epoch)
                writer.add_scalar('test result', test_result, epoch)
                if test_result > max_test_acc:
                    max_test_acc = test_result
                    best_model_state = deepcopy(self.__net.state_dict())
                if verbose and (epoch + 1) % 10 == 0:
                    print('Epoch {:03}. Loss: {:.4f}. Train accuracy: {:.4f}. Test accuracy: {:.4f}.'.
                          format(epoch, total_loss, train_result, test_result))

            if self.__training_params['optimization_params']['save_model']:
                model_state_dict = self.__net.state_dict()
                checkpoint = {"model": model_state_dict, "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "global_step": global_step}
                model_path = os.path.join(self.__training_params['optimization_params']['save_dir'], 'model.pt')
                print('Saving model to {}.{}'.format(model_path, epoch))
                torch.save(checkpoint, '{}.{}'.format(model_path, epoch))
                keep_epochs = 10
                if os.path.isfile('{}.{}'.format(model_path, epoch-keep_epochs)):
                    os.system('rm {}.{}'.format(model_path, epoch-keep_epochs))

        toc = time.perf_counter()
        print('Training completed (time elapsed: {:.4f} s). '.format(toc - tic))

        self.__net.load_state_dict(best_model_state)

        if val_loader is not None:
            tic = time.perf_counter()
            test_result = self.test(test_loader)
            toc = time.perf_counter()
            print('Testing completed (time elapsed: {:.4f} s). '.format(toc - tic))
            print('Best validation accuracy: {:.4f}, corresponding test accuracy: {:.4f}.'.
                  format(max_val_acc, test_result))
            return self.__net, (max_val_acc, test_result)
        else:
            print('Best test accuracy: {:.4f}.'.format(max_test_acc))
            return self.__net, max_test_acc

    def test(self, data_loader, is_train=False, is_validation=False):
        self.__net.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        correct = 0
        for data in data_loader:
            with torch.no_grad():
                if self.__training_params['network_params']['voting_method'] == 'majority':
                    pred = []
                    for _ in range(self.__training_params['network_params']['voting_runs']):
                        pred.append(self.__net(data.to(device))[0])
                    pred = torch.stack(pred, dim=2)
                    pred = pred.sum(dim=2)
                elif self.__training_params['network_params']['voting_method'] == 'entropy':
                    pred = []
                    for _ in range(self.__training_params['network_params']['voting_runs']):
                        pred.append(self.__net(data.to(device))[0])
                    pred = torch.stack(pred, dim=1)
                    
                    #print("pred before:", pred.cpu().numpy())
                    
                    pred_clipped = torch.maximum(pred, torch.ones_like(pred)*1e-37)
                    entr = -torch.sum(pred_clipped * torch.log(pred_clipped), dim = 2)
                    
                    #print("entropy:", entr.cpu().numpy())
                    
                    choice = entr.argmin(dim=1)
                    
                    pred = torch.stack([p[i] for p,i in zip(pred,choice)])
                    #print("pred after:", pred.cpu().numpy())
                else:        
                    pred, loss = self.__net(data.to(device))
                    
                pred = pred.argmax(dim=1)
                    
                label = data.y

            correct += pred.eq(label).sum().item()

        
        total = len(data_loader.dataset)

        return correct / total


def print_log(string, file):
    print(string)
    print(string, file=file)

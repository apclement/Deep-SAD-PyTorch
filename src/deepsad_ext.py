
from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url

import os
import torch
import numpy as np
import pandas as pd

from awsio.python.lib.io.s3.s3dataset import S3Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import io
from torch.utils.data import IterableDataset
from awsio.python.lib.io.s3.s3dataset import S3IterableDataset, ShuffleDataset, S3BaseClass

from itertools import chain
import random
import _pywrap_s3_io

import timeit
import time

from params import *

os.environ['AWS_REGION'] = 'eu-west-3'
os.environ["AWS_ACCESS_KEY_ID"] = "AKIAXZV2S5AJXG27RMGP"
os.environ["AWS_SECRET_ACCESS_KEY"] = "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"
os.environ['AWS_DEFAULT_REGION'] = 'eu-west-3'

class S3_CSV(IterableDataset):
    def __init__(self, urls, shuffle_urls=False, transform=None):
        self.rank = int(os.getenv("RANK", "1"))
        self.world_size = int(os.getenv('WORLD_SIZE', "1"))
        self.transform = transform
        self.shuffle = shuffle_urls # if we shuffle urls, we implicitly want to shuffle data     
        #print(urls)
        #print(f"Chunkify URL list among workers : rank = {self.rank}; world_size = {self.world_size}")
                
        self.s3_iter_dataset = S3IterableDataset(urls, shuffle_urls)
        # manually set for MPI :(
        self.s3_iter_dataset.rank = self.rank
        self.s3_iter_dataset.world_size = self.world_size
        
        # get resolved urls list, keep only csv files
        url_list = [u for u in self.s3_iter_dataset.urls_list if u.endswith('.csv')]
        #print(f"URL list size = {len(url_list)}")
        self.s3_iter_dataset._urls_list = url_list
        
        worker_url_list = self.s3_iter_dataset.worker_dist(self.s3_iter_dataset.shuffled_list)
        print(f"worker url list (chunk size: {len(worker_url_list)}) List will be splitted further among dataloaders :")
                      
         
    def data_generator(self):
        try:
            while True:
                # Based on alphabetical order of files (partitions)
       
                start = timeit.default_timer()
                name, obj = next(self.s3_iter_dataset_iterator)
                epoch = self.s3_iter_dataset.epoch
                #print(f">>> parse csv {name} !")

                # parse csv with pandas and shuffe data
                df = pd.read_csv(io.BytesIO(obj))                
                if self.shuffle:
                    df = df.sample(frac=1, random_state=epoch).reset_index(drop=True)
                    
                #print(df.groupby('label').size())
                              
                # start at 4 index to skip project_hash, ecriture_id, label and target columns and get the feature columns                   
                samples = torch.tensor(df.iloc[:, 4:].to_numpy(), dtype=torch.float32)   
                df.drop(columns=df.columns[4:], inplace=True)
                                
                end = timeit.default_timer()   
                #print(f"Time taken is {end - start}s")                
                
                weights = torch.ones(1, dtype=torch.float32)
                
                for i, row in df.iterrows():   
                    yield samples[i], int(row.label), int(row.target), weights[0], (row.project_hash, row.ecriture_id)

        except StopIteration:
            return
            
    def __iter__(self):
        self.s3_iter_dataset_iterator = iter(self.s3_iter_dataset)
        return self.data_generator()
        
    def set_epoch(self, epoch):
        self.s3_iter_dataset.set_epoch(epoch)

# urls can be a S3 prefix containing all the shards or a list of S3 paths for all the shards 
from torch.utils.data import DataLoader, Subset
from base.base_dataset import BaseADDataset

from base.odds_dataset import ODDSDataset

import torch

class MyADDataset(BaseADDataset):

    def __init__(self, root):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)
        self.known_outlier_classes = (1,)
        
        print(f">>>> FOLD = {FOLD}")
        print(f"train_sample_pct = {train_sample_pct}")

        # Get train set
        self.train_set = S3_CSV(f's3://ml4ra/_vat_training/sagemaker/folds/{FOLD}/train_sample{train_sample_pct}.csv/', shuffle_urls=True)
       
        # Get test set
        self.test_set = S3_CSV(f's3://ml4ra/_vat_training/sagemaker/folds/{FOLD}/test.csv/', shuffle_urls=False)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 1) -> (DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, num_workers=num_workers, drop_last=False)
        return train_loader, test_loader
    
    def set_epoch(self, epoch):
        self.train_set.set_epoch(epoch)

    
from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score

import logging
import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch.distributed as dist


def _average_gradients(model):
        # Gradient averaging.
        size = float(dist.get_world_size())
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size
            
class MyAETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_loss = 0.0
        
        self.train_losses = []
        self.test_losses = []
        self.test_avg_precs = []

    def train(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        print(f">>> Batch size = {self.batch_size}")
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        logger.info('LR scheduler: actuel learning rate is %g' % float(scheduler.get_last_lr()[0]))                   

        # Training
        logger.info('Starting pretraining...')
        start_time = time.time()
        
        dataset.set_epoch(0)        
        n_batches = 0
        for data in train_loader:
            n_batches += 1
        print(f"N batches = {n_batches}")
        
        for epoch in range(self.n_epochs):
            ae_net.train()
            dataset.set_epoch(epoch)
            #print(f"Starting epoch {epoch+1}...")            
             
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:               
                inputs, _, _, _, _ = data
                inputs = inputs.to(self.device)
                
                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                loss = torch.mean(rec_loss)
                loss.backward()
                #_average_gradients(ae_net)
                optimizer.step()               

                epoch_loss += loss.item()
                n_batches += 1      
                if n_batches >= max_batches:
                    break
                    
            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))
                        
            # log epoch statistics
            #print(f"Computing epoch {epoch+1} metrics...")
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')   
            
            if (epoch + 1) % 10 == 0:
                self.test(dataset, ae_net.module)
                
            self.train_losses += [epoch_loss / n_batches]
            self.test_losses += [self.test_loss]           
            
           
        self.train_time = time.time() - start_time
        logger.info('Pretraining Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset: BaseADDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device for network
        ae_net = ae_net.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        #logger.info('Testing autoencoder...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:              
                inputs, labels, _, _, idx = data
                                     
                inputs, labels, idx = inputs.to(self.device), labels.to(self.device), idx #.to(self.device)

                rec = ae_net(inputs)
                rec_loss = criterion(rec, inputs)
                scores = torch.mean(rec_loss, dim=tuple(range(1, rec.dim())))

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss = torch.mean(rec_loss)
                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
       
        # Compute AUC
        labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        #print("Labels test distribution :")       
        #print(pd.Series(labels).value_counts())
                  
        self.test_auc = roc_auc_score(labels, scores)
        self.test_loss = epoch_loss / n_batches
        
        # Log results
        logger.info('Test Loss: {:.6f}'.format(self.test_loss) +' | Test AUC: {:.2f}%'.format(100. * self.test_auc) +' | Test Time: {:.3f}s'.format(self.test_time))
        #logger.info('Finished testing autoencoder.')


class MyDeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_loss = 0
        self.test_auc = None
        self.test_avg_prec = None
        self.test_precision = None
        self.test_recall = None
        self.test_thresholds = None
        self.test_time = None
        self.test_scores = None
        self.test_outputs = None
        
        self.train_losses = []
        self.test_losses = []
        self.test_avg_precs = []
                
    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        logger.info('LR scheduler: actuel learning rate is %g' % float(scheduler.get_last_lr()[0]))                   


        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            dataset.set_epoch(0)   
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')
            
        # Training
        logger.info('Starting training...')
        start_time = time.time()
        
        for epoch in range(self.n_epochs):
            dataset.set_epoch(epoch)                 
            net.train()
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
                        
            for data in train_loader:
                inputs, _, semi_targets, w, _ = data
                inputs, semi_targets, w = inputs.to(self.device), semi_targets.to(self.device), w.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                # eta = torch.where(semi_targets == -1, 300, 1) 
                losses = torch.where(semi_targets == 0, dist, w * self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                loss.backward()
                # manually average gradient 
                #_average_gradients(net)
                optimizer.step()                
                
                epoch_loss += loss.item()
                n_batches += 1
                if n_batches >= max_batches:
                    break
                    
            scheduler.step()            
            if epoch in self.lr_milestones:
                logger.info('LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')          
            
            if (epoch+1) % 10 == 0:
                self.test(dataset, net.module)
                
            self.train_losses += [epoch_loss / n_batches]
            self.test_losses += [self.test_loss]
            self.test_avg_precs += [self.test_avg_prec]
            
        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        #logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        test_outputs = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, _, idx = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                projects, indices = idx

                outputs = net(inputs)                
                dist = torch.sum((outputs - self.c) ** 2, dim=1)               
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(projects.numpy().tolist(),
                                            indices.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))
                #test_outputs += [outputs.cpu().data.numpy()]

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score
        #self.test_outputs = test_outputs

        # Compute AUC
        _, _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        #print(pd.Series(labels).value_counts())        
      
        self.test_auc = roc_auc_score(labels, scores)
        self.test_loss = epoch_loss / n_batches
        self.test_precision, self.test_recall, self.test_thresholds = precision_recall_curve(labels, scores)
        self.test_avg_prec = average_precision_score(labels, scores)

        # Log results
        logger.info(f'Test Loss: {self.test_loss:.6f} | Test AUC: {self.test_auc * 100:.2f}% | Test Avg Precision: {self.test_avg_prec:.3f} | Test Time: {self.test_time:.3f}s')
        #logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        # FIXME: !!!!!! rep dim is set to 2 !!!!
        c = torch.zeros(rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _, _ = data
                inputs = inputs.to(self.device)                
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
    
    
import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.DeepSAD_trainer import DeepSADTrainer
from optim.ae_trainer import AETrainer
from networks.mlp import MLP, MLP_Autoencoder


class MyDeepSAD(object):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, eta: float = 1.0, device = 'cuda'):
        """Inits DeepSAD with hyperparameter eta."""
        
        self.is_distributed = int(os.getenv('WORLD_SIZE', "1")) > 1
        self.rank = int(os.getenv('RANK', "0"))

        self.eta = eta
        self.device = device
        self.c = None  # hypersphere center c

        self.net_name = None
        print(f"X dim = {N_features}")
        self.net = MLP(x_dim=N_features, h_dims=[64, 32, 16, 8, 4], rep_dim=2, bias=False)
         # Set device for network
        self.net = self.net.to(self.device)         
        

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

   
    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 88,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SAD model on the training data."""
        print(f"Start training ...")
        
        if self.is_distributed:            
            # multi-machine multi-gpu case
            print(f"Distributed training - {self.is_distributed}")
            self.net = torch.nn.parallel.DistributedDataParallel(self.net)
        
        self.optimizer_name = optimizer_name
        self.trainer = MyDeepSADTrainer(self.c, self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        if self.trainer is None:
            self.trainer = MyDeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        # unwrap distributed model to test
        self.trainer.test(dataset, self.net.module)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_avg_prec'] = self.trainer.test_avg_prec
        self.results['test_precision'] = self.trainer.test_precision.tolist()
        self.results['test_recall'] = self.trainer.test_recall.tolist()
        self.results['test_thresholds'] = self.trainer.test_thresholds.tolist()
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 88,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        # Set autoencoder network
        self.ae_net = MLP_Autoencoder(x_dim=N_features, h_dims=[64, 32, 16, 8, 4], rep_dim=2, bias=False)
         # Set device for network
        self.ae_net = self.ae_net.to(self.device)
        
        if self.is_distributed:            
            # multi-machine multi-gpu case
            print(f"Distributed training - {self.is_distributed}")
            self.ae_net = torch.nn.parallel.DistributedDataParallel(self.ae_net)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = MyAETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_trainer.train(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        if self.rank == 0:
            print("start ae testing")
            self.ae_trainer.test(dataset, self.ae_net.module)
            print("end ae testing")

        # Get test results
        self.ae_results['test_auc'] = 0#self.ae_trainer.test_auc
        self.ae_results['test_time'] = 0#self.ae_trainer.test_time
        
        #torch.save(self.ae_net.cpu().state_dict(), 'ae_model.pth')
        print("end pretraining")
        
        # Initialize Deep SAD network weights from pre-trained encoder       
        self.init_network_weights_from_pretraining()
        
    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""
        print("Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder.")

        net_dict = self.net.cpu().state_dict()
        ae_net_dict = self.ae_net.cpu().state_dict()
       
        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)
        self.net = self.net.to(self.device) 
        

    def save_model(self, export_model, save_ae=True):
        """Save Deep SAD model to export_model."""

        net_dict = self.net.state_dict()
        # ae_net_dict = self.ae_net.state_dict() if save_ae else None
        ae_net_dict = None

        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        # load autoencoder parameters if specified
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp)

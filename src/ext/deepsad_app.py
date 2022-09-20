
import click
import torch
import logging
import random
import numpy as np
import pandas as pd
import os
from utils.config import Config
from DeepSAD import DeepSAD
from datasets.main import load_dataset
from sklearn.metrics import roc_auc_score

from deepsad_ext import *
from torchdata.datapipes.iter import IterableWrapper, S3FileLoader

import argparse
import torch.distributed as dist

def main():
    """
    Deep SAD, a method for deep semi-supervised anomaly detection.
    """
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    #parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    #parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
   # parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    args, _ = parser.parse_known_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log_file = './log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # distributed ?
    is_distributed = len(args.hosts) > 1 #and args.backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))
    
    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        backend='gloo'
        dist.init_process_group(backend=backend, rank=host_rank, world_size=world_size)
        logger.info(f"Initialized the distributed environment: '{backend}' backend on {dist.get_world_size()} nodes.")
        logger.info(f"Current host rank is {dist.get_rank()}. Number of gpus: {args.num_gpus}")  
        
    host_rank = int(os.getenv("RANK", 0))
           
    seed=42
    eta=1.0
    
    ae_optimizer_name='adam'
    ae_lr=0.1
    ae_n_epochs=60
    ae_lr_milestone=[20, 40, 80]
    ae_batch_size=128
    ae_weight_decay=1e-6
    
    optimizer_name='adam'
    lr=0.01
    n_epochs=50
    #lr_milestone=[20, 80, 120]
    lr_milestone=[20]
    batch_size=128
    weight_decay=1e-6
    
    num_threads = 4
    n_jobs_dataloader = 4
    pretrain = True
    xp_path = '.'
    device = 'cuda'

    # Get configuration
    cfg = Config(locals().copy())

    # Print paths
    logger.info('Log file is %s' % log_file)
    # Print model configuration
    logger.info('Eta-parameter: %.2f' % cfg.settings['eta'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

   
    # Load data
    dataset = MyADDataset(xp_path)

    # Initialize DeepSAD model and set neural network phi
    deepSAD = MyDeepSAD(cfg.settings['eta'], device)
   
    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deepSAD.pretrain(dataset,
                         optimizer_name=cfg.settings['ae_optimizer_name'],
                         lr=cfg.settings['ae_lr'],
                         n_epochs=cfg.settings['ae_n_epochs'],
                         lr_milestones=cfg.settings['ae_lr_milestone'],
                         batch_size=cfg.settings['ae_batch_size'],
                         weight_decay=cfg.settings['ae_weight_decay'],
                         device=device,
                         n_jobs_dataloader=n_jobs_dataloader)

        # Save pretraining results
        #deepSAD.save_ae_results(export_json='./ae_results.json')
        pretrain_auc = deepSAD.ae_results['test_auc']
       
    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    deepSAD.train(dataset,
                  optimizer_name=cfg.settings['optimizer_name'],
                  lr=cfg.settings['lr'],
                  n_epochs=cfg.settings['n_epochs'],
                  lr_milestones=cfg.settings['lr_milestone'],
                  batch_size=cfg.settings['batch_size'],
                  weight_decay=cfg.settings['weight_decay'],
                  device=device,
                  n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deepSAD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Generate result DataFrame
    test_auc = deepSAD.results['test_auc']
    projects, indices, labels, scores = zip(*deepSAD.results['test_scores'])
    
    result_df = pd.DataFrame({'projects': projects, 'indices': indices, 'labels': labels, 'scores': scores})
    result_df.to_csv(f"s3://ml4ra/_vat_training/sagemaker/vat/test_score_labels_{host_rank}.csv", index=False)
    
    def export_metrics(trainer, name):
        # save losses
        n_epochs = len(trainer.train_losses)
        train_losses = pd.DataFrame({'epoch': range(n_epochs),'value': trainer.train_losses, 'metric': 'Train loss'})
        test_losses = pd.DataFrame({'epoch': range(n_epochs),'value': trainer.test_losses, 'metric': 'Test loss'})
        if name == 'sad':
            avg_precs = pd.DataFrame({'epoch': range(n_epochs),'value': trainer.test_avg_precs, 'metric': 'Avg prec.'})
            metrics = pd.concat([train_losses, test_losses, avg_precs])
        else:
            metrics = pd.concat([train_losses, test_losses])

        metrics.to_csv(f"s3://ml4ra/_vat_training/sagemaker/vat/{name}_metrics_{host_rank}.csv", index=False)
        
    export_metrics(deepSAD.ae_trainer, 'ae')
    export_metrics(deepSAD.trainer, 'sad')     
    
    if host_rank == 0:
        logger.info(f">>>> Saving the model to {args.model_dir}.")
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        torch.save({'c': deepSAD.c, 'state': deepSAD.net.cpu().state_dict()}, args.model_dir + '/model.pth')

if __name__ == '__main__':
    main()

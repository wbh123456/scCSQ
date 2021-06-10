import torch
from torchvision import models
from torch import nn
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import os
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import Counter
import statistics
from torchvision.datasets.utils import download_and_extract_archive
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score
# from ray.tune.integration.pytorch_lightning import TuneReportCallback
# from ray import tune
# from ray.tune import CLIReporter
import shutil
import fairscale
import argparse

# Make tqdm module to print in a 1 line
# from functools import partial
# tqdm = partial(tqdm, position=0, leave=True)

from util import *
from dataModule import *


def get_class_balance_loss_weight(samples_in_each_class, n_class, beta=0.9999):
    # Class-Balanced Loss on Effective Number of Samples
    # Reference Paper https://arxiv.org/abs/1901.05555
    weight = (1 - beta)/(1 - torch.pow(beta, samples_in_each_class))
    weight = weight / weight.sum() * n_class
    return weight


def test_class_balance_loss():
    print("Testing class balance loss...")
    # Sample 1
    samples_in_each_class, n_class = torch.tensor([15, 10, 10, 10, 19]), 5
    corr1 = torch.tensor(
        [0.79511815, 1.1923454, 1.1923454, 1.1923454, 0.6278458])
    ans1 = get_class_balance_loss_weight(samples_in_each_class, n_class)
    if np.array_equal(corr1.numpy(), ans1.numpy()):
        print("Test 1 passes")
    else:
        print("Test1 failed", "ans1 =", ans1.numpy(),
              "correct1 =", corr1.numpy())

    # Sample 2
    samples_in_each_class, n_class = torch.tensor([1, 1, 1, 1, 1]), 5
    corr2 = torch.tensor([1., 1., 1., 1., 1.])
    ans2 = get_class_balance_loss_weight(samples_in_each_class, n_class)
    if np.array_equal(corr2.numpy(), ans2.numpy()):
        print("Test 2 passes")
    else:
        print("Test2 failed", "ans2 =", ans2.numpy(),
              "correct2 =", corr2.numpy())

    # Sample 3
    samples_in_each_class, n_class = torch.tensor([1, 2, 4, 8, 16]), 5
    corr3 = torch.tensor(
        [2.5801828, 1.2904761, 0.64523804, 0.32269114, 0.16141175])
    ans3 = get_class_balance_loss_weight(samples_in_each_class, n_class)
    if np.array_equal(corr3.numpy(), ans3.numpy()):
        print("Test 3 passes")
    else:
        print("Test3 failed", "ans3 =", ans3.numpy(),
              "correct3 =", corr3.numpy())

    return

###------------------------------Model---------------------------------------###


class CSQLightening(pl.LightningModule):
    def __init__(self, n_class, n_features, batch_size=64, l_r=1e-5, lamb_da=0.0001, beta=0.9999, bit=64, lr_decay=0.9, decay_every=20):
        super(CSQLightening, self).__init__()
        print("hparam: l_r = {}, lambda = {}, beta = {}".format(l_r, lamb_da, beta))
        self.batch_size = batch_size
        self.l_r = l_r
        self.bit = bit
        self.n_class = n_class
        self.lamb_da = lamb_da
        self.beta = beta
        self.lr_decay = lr_decay
        self.decay_every = decay_every
        self.samples_in_each_class = None  # Later initialized in training step
        # self.hash_centers = get_hash_centers(self.n_class, self.bit)
        ##### model structure ####
        self.hash_layer = nn.Sequential(
            nn.Linear(n_features, 9000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(9000, 3150),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(3150, 900),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(900, 450),
            nn.ReLU(inplace=True),
            nn.Linear(450, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, self.bit),
        )

        self.center_layer = nn.Sequential(
            nn.Linear(1,10),
            nn.ReLU(inplace=True),
            nn.Linear(10,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,n_class*self.bit),
        )

    def forward(self, x):
        # forward pass returns prediction
        x = self.hash_layer(x)
        return x

    def forward_hash_center(self):
        # Fixed input 1. Hardcoded the input on GPU
        one_input = torch.from_numpy(np.array([1])).cuda().float()
        learned_hash_centers = self.center_layer(one_input)
        # Reshape the output to (# of classes) x (# of bits)
        learned_hash_centers = torch.reshape(learned_hash_centers, (self.n_class, self.bit))
        return learned_hash_centers

    def get_class_balance_loss_weight(samples_in_each_class, n_class, beta=0.9999):
        # Class-Balanced Loss on Effective Number of Samples
        # Reference Paper https://arxiv.org/abs/1901.05555
        weight = (1 - beta)/(1 - torch.pow(beta, samples_in_each_class))
        weight = weight / weight.sum() * n_class
        return weight

    def get_closest_false_hash_center(self, centers, labels, hash_codes):
        '''
        centers: learned hashing center: C * K
        labels: N
        hash_codes: N * K
        '''
        # Get distances to the closest hash center that is not the label of the input
        distances = euclidean_distance(hash_codes, centers)
        # print("dis = ", distances)
        mask = make_one_hot(labels, self.n_class)
        mask = mask.cuda()

        valmax = distances.max()

        # Add valmax to distances of correct hashing centers
        # Trick to not select the correct hashing center in min() operation
        distances_incor = distances + mask * valmax
        distances_cor = distances + (1-mask) * valmax

        dist_cor, _ = distances_cor.min(-1)
        min_dist_incor, _ = distances_incor.min(-1)

        loss = F.relu(dist_cor - min_dist_incor + 32)
        loss = loss.mean()
        # print("LOSS=", loss)
        return loss

    def CSQ_loss_function(self, hash_codes, learned_hash_centers, labels):
        hash_codes = hash_codes.tanh()
        if learned_hash_centers != None:
            # Learning hashing centers
            hash_centers = learned_hash_centers[labels]
        else:
            # Using fixed hashing centers
            hash_centers = self.hash_centers[labels]
            hash_centers = hash_centers.type_as(hash_codes)

        if self.samples_in_each_class == None:
            self.samples_in_each_class = self.trainer.datamodule.samples_in_each_class
            self.n_class = self.trainer.datamodule.N_CLASS

        weight = get_class_balance_loss_weight(
            self.samples_in_each_class, self.n_class, self.beta)
        weight = weight[labels]
        weight = weight.type_as(hash_codes)

        # Center Similarity Loss
        BCELoss = nn.BCELoss(weight=weight.unsqueeze(1).repeat(1, self.bit))
        # BCELoss = nn.BCELoss()
        C_loss = BCELoss(0.5 * (hash_codes + 1),
                         0.5 * (hash_centers.detach() + 1)) # Detach hash centers to avoid bug, so we don't train hash centers here

        # Center Loss
        center_loss = self.get_closest_false_hash_center(learned_hash_centers, labels, hash_codes)
        
        # Quantization Loss
        Q_loss = (hash_codes.abs() - 1).pow(2).mean()

        loss = C_loss + self.lamb_da * Q_loss + 0.9*center_loss
        return loss

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch
        hash_codes = self.forward(data)
        learned_hash_centers = self.forward_hash_center()
        loss = self.CSQ_loss_function(hash_codes, learned_hash_centers, labels)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        hash_codes = self.forward(data)
        learned_hash_centers = self.forward_hash_center()
        loss = self.CSQ_loss_function(hash_codes, learned_hash_centers, labels)
        return loss

    def validation_epoch_end(self, outputs):

        val_loss_epoch = torch.stack([x for x in outputs]).mean()

        database_dataloader = self.trainer.datamodule.database_dataloader
        val_dataloader = self.trainer.datamodule.val_dataloader()

        val_matrics_CHC = compute_metrics(val_dataloader, self, self.n_class)
        val_labeling_accuracy_CHC, val_F1_score_weighted_average_CHC, val_F1_score_median_CHC, val_F1_score_per_class_CHC = val_matrics_CHC

        if not self.trainer.running_sanity_check:
            print(f"Epoch: {self.current_epoch}, Val_loss_epoch: {val_loss_epoch:.2f}")
            print(f"val_F1_score_median_CHC:{val_F1_score_median_CHC:.3f}, val_labeling_accuracy_CHC:{val_labeling_accuracy_CHC:.3f},\
                   val_F1_score_weighted_average_CHC:{val_F1_score_weighted_average_CHC:.3f},\
                   val_F1_score_per_class_CHC:{[f'{score:.3f}' for score in val_F1_score_per_class_CHC]}")

        value = {"Val_loss_epoch": val_loss_epoch, 
                  "Val_F1_score_median_CHC_epoch": val_F1_score_median_CHC,
                  "Val_labeling_accuracy_CHC_epoch": val_labeling_accuracy_CHC, 
                  "Val_F1_score_weighted_average_CHC_epoch": val_F1_score_weighted_average_CHC, }
        self.log_dict(value, prog_bar=True, logger=True)

    def test_step(self, test_batch, batch_idx):
        data, labels = test_batch
        hash_codes = self.forward(data)
        learned_hash_centers = self.forward_hash_center()
        loss = self.CSQ_loss_function(hash_codes, learned_hash_centers, labels)
        return loss

    def test_epoch_end(self, outputs):
        test_loss_epoch = torch.stack([x for x in outputs]).mean()

        database_dataloader = self.trainer.datamodule.database_dataloader
        test_dataloader = self.trainer.datamodule.test_dataloader()

        test_matrics_CHC = compute_metrics(test_dataloader, self, self.n_class, show_time=True, use_cpu=True)
        test_labeling_accuracy_CHC, test_F1_score_weighted_average_CHC, test_F1_score_median_CHC, test_F1_score_per_class_CHC = test_matrics_CHC
        
        # Test speed
        test_speed(test_dataloader, self, 200)
        test_speed(test_dataloader, self, 500)
        test_speed(test_dataloader, self, 1000)
        test_speed(test_dataloader, self, 5000)
        test_speed(test_dataloader, self, 10000)

        if not self.trainer.running_sanity_check:
            print(f"Epoch: {self.current_epoch}, Test_loss_epoch: {test_loss_epoch:.2f}")
            print(f"test_F1_score_median_CHC:{test_F1_score_median_CHC:.3f}, test_labeling_accuracy_CHC:{test_labeling_accuracy_CHC:.3f}, \
                    test_F1_score_weighted_average_CHC:{test_F1_score_weighted_average_CHC:.3f}, \
                    test_F1_score_per_class_CHC:{[f'{score:.3f}' for score in test_F1_score_per_class_CHC]}")

        value = {"Test_loss_epoch": test_loss_epoch,
                 "Test_F1_score_median_CHC_epoch": test_F1_score_median_CHC,
                 "Test_labeling_accuracy_CHC_epoch": test_labeling_accuracy_CHC, 
                 "Test_F1_score_weighted_average_CHC_epoch": test_F1_score_weighted_average_CHC}
        self.log_dict(value, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # optimizer = torch.optim.RMSprop(self.parameters(),
        #                                 lr=self.l_r, weight_decay=10**-4)
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.l_r, weight_decay=10**-5)

        # Decay LR by a factor of gamma every step_size epochs
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=self.decay_every, gamma=self.lr_decay)

        return [optimizer], [exp_lr_scheduler]


if __name__ == '__main__':
    pl.callbacks.progress.ProgressBar(refresh_rate=1)

    # Parse parameters
    parser = argparse.ArgumentParser()
    # Hyperparameters
    parser.add_argument("--l_r", type=float, default=1.2e-5,
                        help="learning rate")
    parser.add_argument("--lamb", type=float, default=0.001,
                        help="lambda of quantization loss")
    parser.add_argument("--beta", type=float, default=0.9999,
                        help="beta of class balance loss")
    parser.add_argument("--lr_decay", type=float, default=0.5,
                        help="learning rate decay")
    parser.add_argument("--decay_every", type=int, default=25,
                        help="how many epochs a learning rate happens brefore it decays")
    # Training parameters
    parser.add_argument("--epochs", type=int, default=301,
                        help="number of epochs to run")
    parser.add_argument("--dataset", choices=['TM', 'BaronHuman', 'Zheng68K', 'AMB', "XIN"],
                        help="dataset to train against")
    # Control parameters
    parser.add_argument("--test", type=str, default='',
                        help="To test against a specific checkpoint")
    args = parser.parse_args()

    l_r = args.l_r
    lamb_da = args.lamb
    beta = args.beta
    max_epochs = args.epochs
    dataset = args.dataset
    lr_decay = args.lr_decay
    decay_every = args.decay_every
    test_checkpoint = args.test

    print(args)

    # set up datamodule
    if dataset == "TM":
        datamodule = TMDataModule(import_size=1, num_workers=4)
        N_CLASS = 55
        N_FEATURES = datamodule.N_FEATURES
    elif dataset == "BaronHuman":
        datamodule = BaronHumanDataModule(num_workers=4, batch_size=128)
        N_CLASS = 13
        N_FEATURES = 17499
    elif dataset == "Zheng68K":
        datamodule = Zheng68KDataModule(num_workers=4)
        N_CLASS = 11
        N_FEATURES = 20387
    elif dataset == "AMB":
        # annotation_level可以是3，16或者92
        datamodule = AMBDataModule(num_workers=4, annotation_level=92)
        N_CLASS = 93
        N_FEATURES = 42625
    elif dataset == "XIN":
        datamodule = XinDataModule(num_workers=4, batch_size=128)
        N_CLASS = 4
        N_FEATURES = 33889
    else:
        print("Unknown dataset:", dataset)
        exit()

    # Init ModelCheckpoint callback
    checkpointPath = "checkpoints/" + dataset
    checkpoint_callback = ModelCheckpoint(monitor='Val_F1_score_median_CHC_epoch',
                                          dirpath=checkpointPath,
                                          filename='CSQ-{epoch:02d}-{Val_F1_score_median_CHC_epoch:.3f}',
                                          verbose=True,
                                          mode='max')


    # To test against a specific checkpoint
    if test_checkpoint != '':
        trainer = pl.Trainer(max_epochs=max_epochs,
                        gpus=0,
                        check_val_every_n_epoch=5,
                        checkpoint_callback=checkpoint_callback
                        )
        model = CSQLightening.load_from_checkpoint(
            test_checkpoint, n_class=N_CLASS, n_features=N_FEATURES)
        model.eval()

        trainer.test(model, datamodule=datamodule)

    # Train
    else:
        trainer = pl.Trainer(max_epochs=max_epochs,
                            gpus=1,
                            check_val_every_n_epoch=5,
                            #  limit_train_batches=0.2,
                            #  limit_val_batches=0.2,
                            #  accelerator='ddp',
                            #  plugins='ddp_sharded',
                            # checkpoint_callback=checkpoint_callback
                            )
        model = CSQLightening(N_CLASS, N_FEATURES, l_r=l_r, lamb_da=lamb_da,
                              beta=beta, lr_decay=lr_decay, decay_every=decay_every)

        trainer.fit(model, datamodule)
        trainer.test(model)

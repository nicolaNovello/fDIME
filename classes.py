from __future__ import print_function, division

import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from torch import nn
import torch.nn.functional as F
from datetime import datetime
from utils import *
import time

import numpy as np
import scipy.io as sio


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim):
        '''
        Initialize the discriminator.
        '''
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, input_tensor):
        output_tensor = self.main(input_tensor)
        return output_tensor


class CombinedArchitecture(nn.Module):
    """
    Class combining two equal neural network architectures.
    """
    def __init__(self, single_architecture, divergence):
        super(CombinedArchitecture, self).__init__()
        self.divergence = divergence
        self.single_architecture = single_architecture
        if self.divergence == "GAN":
            self.final_activation = nn.Sigmoid()
        elif self.divergence == "KL" or self.divergence == "HD":
            self.final_activation = nn.Softplus()
        else:
            self.final_activation = nn.Identity()

    def forward(self, input_tensor_1, input_tensor_2):
        intermediate_1 = self.single_architecture(input_tensor_1)
        output_tensor_1 = self.final_activation(intermediate_1)
        intermediate_2 = self.single_architecture(input_tensor_2)
        output_tensor_2 = self.final_activation(intermediate_2)
        return output_tensor_1, output_tensor_2


class ConcatCritic(nn.Module):
    """Concat critic, where the inputs are concatenated and reshaped in a squared matrix."""
    def __init__(self, dim, hidden_dim, layers, activation, divergence):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(dim * 2, hidden_dim, 1, layers, activation)
        if divergence == "GAN":
            self.last_activation = nn.Sigmoid()
        elif divergence == "KL" or divergence == "HD":
            self.last_activation = nn.Softplus()
        else:
            self.last_activation = nn.Identity()

    def forward(self, x, y):
        batch_size = x.size(0)
        # Create all the possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [batch_size * batch_size, -1])
        scores = self._f(xy_pairs)
        out = torch.reshape(scores, [batch_size, batch_size]).t()
        out = self.last_activation(out)
        return out


class SeparableCritic(nn.Module):
    """Separable critic. where the output value is the inner product between the outputs of g(x) and h(y). """
    def __init__(self, dim, hidden_dim, embed_dim, layers, activation, divergence):
        super(SeparableCritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation)
        if divergence == "GAN":
            self.last_activation = nn.Sigmoid()
        elif divergence == "KL" or divergence == "HD":
            self.last_activation = nn.Softplus()
        else:
            self.last_activation = nn.Identity()

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return self.last_activation(scores)


class ConvolutionalCritic(nn.Module):
    """Convolutional critic, used for the consistency tests"""
    def __init__(self, divergence, test_type):
        super(ConvolutionalCritic, self).__init__()
        if "bs" in test_type:
            n_ch_input = 2
        else:
            n_ch_input = 4
        self.conv = nn.Sequential(
            nn.Conv2d(n_ch_input, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.lin = nn.Sequential(
            nn.Linear(6272, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        if divergence == "GAN":
            self.last_activation = nn.Sigmoid()
        elif divergence == "KL" or divergence == "HD":
            self.last_activation = nn.Softplus()
        else:
            self.last_activation = nn.Identity()

    def forward(self, x, y):
        batch_size = x.size(0)
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        cat_pairs = torch.cat((x_tiled, y_tiled), dim=2)
        xy_pairs = torch.reshape(cat_pairs, [batch_size * batch_size, -1, 28, 28])
        scores_tmp = self.conv(xy_pairs)
        flattened_scores_tmp = torch.flatten(scores_tmp, start_dim=1)
        out = self.lin(flattened_scores_tmp)
        out = torch.reshape(out, [batch_size, batch_size]).t()
        out = self.last_activation(out)
        return out


class fDIME():
    """ The class fDIME handles all the estimators, not only the f-DIME ones."""
    def __init__(self, EbN0=None, rho=None, divergence='KL', architecture='deranged', latent_dim=2, alpha=1,
                 watch_training=False, test_type=None, device="cpu"):
        # Input shape
        self.latent_dim = latent_dim
        self.architecture = architecture
        self.joint_dim = 2 * self.latent_dim
        self.EbN0 = EbN0
        self.divergence = divergence  # type of f-divergence to use for training and estimation
        self.alpha = alpha
        self.watch_training = watch_training
        self.rho = rho
        self.test_type = test_type
        output_dim = 1  # Mutual information estimate is a scalar
        self.device = device

        # Noise std based on EbN0 in dB
        if self.EbN0 is not None:
            self.eps = np.sqrt(pow(10, -0.1 * self.EbN0) / (2 * 0.5))

        # Build and compile the discriminator
        if self.architecture == 'joint':
            self.discriminator = ConcatCritic(latent_dim, 256, 2, 'relu', self.divergence).to(self.device)
        elif self.architecture == 'separable':
            self.discriminator = SeparableCritic(latent_dim, 256, 32, 2, 'relu', self.divergence).to(self.device)
        elif self.architecture == 'deranged':
            single_model = Discriminator(self.joint_dim, output_dim)
            self.discriminator = CombinedArchitecture(single_model, self.divergence).to(self.device)
        elif self.architecture == 'conv_critic':
            self.discriminator = ConvolutionalCritic(self.divergence, self.test_type).to(self.device)

    def update_SNR(self, SNR):
        self.EbN0 = SNR
        self.eps = np.sqrt(pow(10, -0.1 * self.EbN0) / (2 * 0.5))

    def update_rho(self, rho):
        self.rho = rho

    def train(self, epochs, batch_size=40, cubic=False, verbose=False, lr=5e-4):
        """Train the discriminator"""
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        self.discriminator.train()
        optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)

        fDIME_training = np.zeros((epochs, 1))
        R_training = np.zeros((epochs, 1))
        loss_training = np.zeros((epochs, 1))
        buffer = torch.tensor(1.0)  # .to(device)

        for epoch in range(epochs):
            if self.rho is None:
                x, y = sample_gaussian(batch_size, self.latent_dim, self.eps, cubic=cubic)
            else:
                x, y = sample_correlated_gaussian(dim=self.latent_dim, rho=self.rho, batch_size=batch_size, cubic=cubic, device=self.device)

            optimizer.zero_grad()
            x = torch.Tensor(x).to(self.device)
            y = torch.Tensor(y).to(self.device)
            if "deranged" in self.architecture:
                data_xy, data_x_y = data_generation_mi(x, y, device=self.device)
                D_value_1, D_value_2 = self.discriminator(data_xy, data_x_y)
                loss, R = compute_loss_ratio(self.divergence, self.architecture, D_value_1=D_value_1, D_value_2=D_value_2,
                                             scores=None, buffer=buffer, alpha=self.alpha, device=self.device)
            else:
                scores = self.discriminator(x, y)
                loss, R = compute_loss_ratio(self.divergence, self.architecture, D_value_1=None, D_value_2=None,
                                             scores=scores, buffer=buffer, alpha=self.alpha, device=self.device)
            loss.backward()
            optimizer.step()
            fDIME_e = torch.log(R)
            if verbose and epoch % 1000 == 0:
                # Plot the progress
                print(f"{epoch} [Cubic: {cubic}, Divergence: {self.divergence}, Architecture: {self.architecture}, "
                      f"Total loss : {loss.item():.4f}, MI now: {torch.mean(fDIME_e):.4f}")
            if self.watch_training:
                fDIME_training[epoch] = np.mean(fDIME_e.cpu().detach().numpy())  # save mutual info
                R_training[epoch] = np.mean(R.cpu().detach().numpy())
                loss_training[epoch] = np.mean(loss.cpu().detach().numpy())
        return fDIME_training, R_training, loss_training

    def train_convolutional(self, train_dataset, opt_params, proc_params, architecture, divergence, it_var, test_type):
        """Train the convolutional discriminator for the consistency tests"""
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        optimizer = optim.Adam(self.discriminator.parameters(), lr=opt_params['lr'])
        self.discriminator.train()
        train_dataloader = DataLoader(train_dataset, batch_size=opt_params['batch_size'], shuffle=True)
        fDIME_training = []
        for epoch in range(opt_params['epochs']):
            fDIME_e_batch = []
            for sample_batched in train_dataloader:
                x = sample_batched[0].to(self.device)
                optimizer.zero_grad()
                x, y = obtain_y_consistency(x, test_type, it_var, device=self.device)
                scores = self.discriminator(x, y)
                loss, R = compute_loss_ratio(divergence, architecture, D_value_1=None, D_value_2=None,
                                             scores=scores, buffer=None, alpha=1, device=self.device)
                loss.backward()
                optimizer.step()
                fDIME_e_batch.append(np.mean(torch.log(R).cpu().detach().numpy()))
            if proc_params['verbose']:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(
                    f"Epoch:{epoch} ({current_time}) [Divergence: {self.divergence}, Architecture: {self.architecture}, "
                    f"Test type: {test_type}, It_var: {it_var}, Total loss : {loss.item():.4f}, MI now: {np.mean(fDIME_e_batch):.4f}")
            if proc_params['watch_training']:  # save values every 10 epochs
                fDIME_training.append(np.mean(fDIME_e_batch))
        return fDIME_training

    def test_convolutional(self, test_dataset, architecture, divergence, it_var, test_type):
        """Test the convolutional discriminator"""
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        self.discriminator.eval()
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        with torch.no_grad():
            fDIME_e = []
            for sample_batched in test_dataloader:
                x = sample_batched[0].to(self.device)
                x, y = obtain_y_consistency(x, test_type, it_var, device=self.device)
                scores = self.discriminator(x, y)
                loss, R = compute_loss_ratio(divergence, architecture, D_value_1=None, D_value_2=None, scores=scores,
                                             buffer=None, alpha=1, device=self.device)
                fDIME_e.append(np.mean(torch.log(R).cpu().detach().numpy()))
        return np.mean(fDIME_e)


def consistency_test_step(proc_params, opt_params, architecture, divergence, it_var, test_type, train_dataset,
                          test_dataset, fdime_training_dict, fdime_test_dict, device):
    fdime = fDIME(architecture=architecture, divergence=divergence,
                  watch_training=proc_params['watch_training'], test_type=test_type, device=device)
    fDIME_training = fdime.train_convolutional(train_dataset, opt_params, proc_params, architecture, divergence, it_var, test_type)
    print("Finished training...")
    fDIME_test = fdime.test_convolutional(test_dataset, architecture, divergence, it_var, test_type)
    key_tmp = "{}_{}_{}_{}".format(divergence, architecture, it_var, test_type)
    fdime_training_dict[key_tmp] = fDIME_training
    fdime_test_dict[key_tmp] = fDIME_test
    return fdime_training_dict, fdime_test_dict


def train_staircase(proc_params, data_params, opt_params):
    """Train the neural networks by generating data from a reference staircase of mutual information values"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for latent_dim in data_params['latent_dim']:
        # Save time needed and mutual information estimates
        time_dict = {}
        staircases = {}
        for cubic in [False, True]:
            for divergence in proc_params['divergences']:
                if divergence == "NJEE":  # There is one possible architecture
                    architecture = "ad_hoc"
                    # Run NJEE
                    start_time = time.time()
                    fDIME_training_staircase = njee_staircase(proc_params, opt_params, cubic, latent_dim)
                    end_time = time.time()
                    # Save NJEE results
                    staircases[f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = fDIME_training_staircase
                    time_dict[f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}_{latent_dim}'] = (float(end_time) - float(start_time)) / 60
                else:
                    for architecture in proc_params['architectures']:
                        if architecture == "deranged" and divergence == "CPC":
                            time_dict[f'{cubic}_{divergence}_{architecture}'] = 0
                        else:
                            start_time = time.time()
                            # Where to save the mutual information estimates
                            fDIME_training_staircase = np.zeros((proc_params['tot_len_stairs']))
                            print_training_title(divergence, architecture)
                            if data_params['rho_gauss_corr']:  # Rho formulation
                                rho = mi_to_rho(latent_dim, proc_params['levels_MI'][0])
                                SNR = None
                            else:  # SNR formulation
                                rho = None
                                SNR = 10 * np.log10(np.exp(2 * proc_params['levels_MI'][0] / latent_dim) - 1)

                            fdime = fDIME(EbN0=SNR, rho=rho, divergence=divergence, architecture=architecture,
                                          latent_dim=latent_dim, alpha=proc_params['alpha'],
                                          watch_training=proc_params['watch_training'], device=device)
                            # Train the estimator for all the steps in the staircase
                            for idx, level_MI in enumerate(proc_params['levels_MI']):
                                if data_params['rho_gauss_corr']:
                                    rho = mi_to_rho(latent_dim, level_MI)
                                    fdime.update_rho(rho)
                                else:
                                    SNR = 10 * np.log10(np.exp(2 * level_MI / latent_dim) - 1)
                                    fdime.update_SNR(SNR)
                                # Train
                                fDIME_training, _, _ = fdime.train(epochs=proc_params['len_step'], batch_size=opt_params['batch_size'],
                                                                         cubic=cubic, verbose=True, lr=opt_params['lr'])
                                fDIME_training_staircase[proc_params['len_step'] * idx:proc_params['len_step'] * (idx + 1)] = np.squeeze(fDIME_training)

                            end_time = time.time()
                            time_dict[f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}_{latent_dim}'] = (float(end_time) - float(start_time))/60
                            staircases[f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}'] = fDIME_training_staircase
                            del fdime
                            #plot_staircase(fDIME_training_staircase, proc_params['tot_len_stairs'], divergence, proc_params['len_step'],
                            #               opt_params['batch_size'], "Mutual information [nats]",
                            #               "Results/Stairs/Staircase_{}_{}_d{}_bs{}_cubic{}.svg".format(divergence, architecture,
                            #                                                                            latent_dim,
                            #                                                                            opt_params['batch_size'],
                            #                                                                            cubic), proc_params['levels_MI'])

        # Plot the staircases for all the algorithms
        plot_staircases(staircases, proc_params, opt_params, latent_dim)
        save_time_dict(time_dict, latent_dim, opt_params["batch_size"])


def consistency_test(proc_params, opt_params, help_dict):
    """Run the self-consistency tests"""
    digits = False
    train_dataset, test_dataset, num_features = load_mnist(digits=digits)
    fdime_training_dict = {}
    fdime_test_dict = {}
    for test_type in proc_params['test_types']:
        print("Starting with test type: ", test_type)
        for architecture in proc_params['architectures']:
            print("Starting with architecture: ", architecture)
            for divergence in proc_params['divergences']:
                print("Starting with divergence: ", divergence)
                processing_type = test_type[3:]
                print("Processing type: ", processing_type)
                for it_var in proc_params[help_dict[processing_type]]:
                    fdime_training_dict, fdime_test_dict = consistency_test_step(proc_params, opt_params, architecture, divergence, it_var, test_type,
                                                                                 train_dataset, test_dataset,
                                                                                 fdime_training_dict, fdime_test_dict,
                                                                                 device=proc_params["device"])

        print("fdime_training_dict: ", fdime_training_dict)
        print("fdime_test_dict: ", fdime_test_dict)
        save_cons_test_results(fdime_test_dict, proc_params, test_type, help_dict, digits)



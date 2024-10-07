import numpy as np
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt
from scipy import stats
import scipy
import pandas as pd # this module is useful to work with tabular data
import random # this module will be used to select random samples from a collection
import os # this module will be used just to create directories in the local filesystem
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from torch import nn
from PIL import Image # to interact with images
import torch.nn.functional as F
import time
import scipy.interpolate as interpolate
from math import *
from scipy.integrate import quad
import json
import csv


def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true* y_pred)


def reciprocal_loss(y_true, y_pred):
    return torch.mean(torch.pow(y_true*y_pred,-1))


def my_binary_crossentropy(y_true, y_pred):
    return -torch.mean(torch.log(y_true)+torch.log(y_pred))


def logmeanexp_loss(y_pred, device="cpu"):
    eps = 1e-5
    batch_size = y_pred.size(0)
    logsumexp = torch.logsumexp(y_pred, dim=(0,))
    return logsumexp - torch.log(torch.tensor(batch_size).float() + eps).to(device)


def phi(x, mu, sigma):
    N,D = np.shape(x)
    unif_output = np.zeros((N,D))
    for i in range(N):
        for j in range(D):
            unif_output[i,j] = (1 + erf((x[i,j] - mu) / sigma / sqrt(2))) / 2
    return unif_output


def derangement(l, device):
    """Random derangement"""
    o = l[:]
    while any(x == y for x, y in zip(o, l)):
        random.shuffle(l)
    return torch.Tensor(l).int().to(device)


def data_generation_mi(data_x, data_y, device="cpu"):
    """
    Generates samples of the product of marginal distributions, given the samples from the joint distribution.
    """
    der = True
    data_xy = torch.hstack((data_x, data_y))
    if der:  # Derangement
        data_y_shuffle = torch.index_select(data_y, 0, derangement(list(range(data_y.shape[0])), device))
        #ordered_derangement = [(idx + 1) % data_y.shape[0] for idx in range(data_y.shape[0])]
        #data_y_shuffle = torch.index_select(data_y, 0, torch.Tensor(ordered_derangement).int().to(device))
    else:  # Permutation
        data_y_shuffle = torch.index_select(data_y, 0, torch.tensor(np.random.permutation(data_y.shape[0])).int().to(device))

    data_x_y = torch.hstack((data_x, data_y_shuffle))
    return data_xy, data_x_y


def sample_gaussian(batch_size, latent_dim, eps, mode="gauss"):
    """Generate samples from a correlated Gaussian distribution of the type Y = X + N"""
    x = np.random.normal(0, 1, (batch_size, latent_dim))
    y = x + eps * np.random.normal(0, 1, (batch_size, latent_dim))
    if mode == "cubic":
        y = y**3
    elif mode == "half-cube":
        x = np.power(np.abs(x), 3/2) * np.sign(x)
        y = np.power(np.abs(y), 3 / 2) * np.sign(y)
    elif mode == "asinh":
        x = np.log(x + np.sqrt(1 + np.power(x,2)))
        y = np.log(y + np.sqrt(1 + np.power(y,2)))
    return x, y


def sample_uniform(batch_size, latent_dim, eps):
    x = np.random.uniform(0, 1, (batch_size, latent_dim))
    n = np.random.uniform(-eps, eps, (batch_size, latent_dim))
    y = x + n
    return x, y


def sample_swiss(batch_size, eps, device="cpu"):
    latent_dim = 1
    x = np.random.normal(0, 1, (batch_size, latent_dim))
    y = x + eps * np.random.normal(0, 1, (batch_size, latent_dim))

    data_u = torch.tensor(phi(x, 0, 1)).float().to(device)
    data_v = torch.tensor(phi(y, 0, np.sqrt(1 + eps ** 2))).float().to(device)

    t_x = 3 * torch.pi / 2 * (1 + 2 * data_u)
    e_x_1 = 1 / 21 * t_x * torch.cos(t_x)
    e_x_2 = 1 / 21 * t_x * torch.sin(t_x)
    e_x = torch.cat((e_x_1, e_x_2), dim=1)
    return e_x, data_v

def sample_student(batch_size, latent_dim, rho, df):
    mean_t = np.zeros(2*latent_dim)
    shape_t = np.eye(2*latent_dim, 2*latent_dim)
    xy = stats.multivariate_t.rvs(loc=mean_t, shape=shape_t, df=df, size=batch_size)
    return xy[:, :latent_dim], xy[:, latent_dim:]

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, mode="gauss", device="cpu"):
    """Generate samples from a correlated Gaussian distribution depending on correlation rho."""
    x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps
    if mode == "cubic":
        y = y ** 3
    elif mode == "half-cube":
        x = torch.pow(torch.abs(x), 3/2) * torch.sign(x)
        y = torch.pow(torch.abs(y), 3 / 2) * torch.sign(y)
    elif mode == "asinh":
        x = torch.log(x + torch.sqrt(1 + torch.pow(x,2)))
        y = torch.log(y + torch.sqrt(1 + torch.pow(y,2)))
    return x.to(device), y.to(device)

def sample_distribution(rho_gauss_corr, latent_dim=20, rho=0, eps=0, df=1, batch_size=64, mode="gauss", device="cpu"):
    if mode == "gauss" or mode == "cubic" or mode == "half-cube" or mode == "asinh":
        if rho_gauss_corr:
            x, y = sample_correlated_gaussian(dim=latent_dim, rho=rho, batch_size=batch_size, mode=mode, device=device)
        else:
            x, y = sample_gaussian(batch_size, latent_dim, eps, mode=mode)
    elif mode == "uniform":
        x, y = sample_uniform(batch_size, latent_dim, eps)
    elif mode == "swiss":
        x, y = sample_swiss(batch_size, eps)
    elif mode == "student":
        x, y = sample_student(batch_size, latent_dim, rho, df)
    return x, y


def mi_to_rho(dim, mi):
    """Obtain the rho for Gaussian, given the ground truth mutual information."""
    return np.sqrt(1 - np.exp(-2.0 / dim * mi))

def mlp(dim, hidden_dim, output_dim, layers, activation):
    """Create a mlp"""
    activation = {
        'relu': nn.ReLU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


def compute_loss_ratio(divergence, architecture, device, D_value_1=None, D_value_2=None, scores=None, buffer=None, alpha=1):
    if divergence == 'KL':
        if "deranged" in architecture:
            loss, R = kl_fdime_deranged(D_value_1, D_value_2, alpha=alpha, device=device)
        else:
            loss, R = kl_fdime_e(scores, device=device)

    elif divergence == 'GAN':
        if "deranged" in architecture:
            loss, R = gan_fdime_deranged(D_value_1, D_value_2, device=device)
        else:
            loss, R = gan_fdime_e(scores, device=device)

    elif divergence == 'HD':
        if "deranged" in architecture:
            loss, R = hd_fdime_deranged(D_value_1, D_value_2, device=device)
        else:
            loss, R = hd_fdime_e(scores, device=device)

    elif divergence == "RKL":
        if "deranged" in architecture:
            loss, R = rkl_fdime_deranged(D_value_1, D_value_2, device=device)
        else:
            loss, R = rkl_fdime_e(scores, device=device)

    elif divergence == 'MINE':
        if "deranged" in architecture:
            loss, R, buffer = mine_ma_deranged(D_value_1, D_value_2, buffer)
        else:
            loss, R, buffer = mine_ma(scores, buffer, momentum=0.9, device=device)

    elif divergence == 'SMILE':
        tau = 1.0 # np.inf
        if "deranged" in architecture:
            loss, R = smile_deranged(D_value_1, D_value_2, tau, device=device)
        else:
            loss, R = smile(scores, clip=tau, device=device)

    elif divergence == "CPC":
        if "deranged" in architecture:
            loss = torch.Tensor(0)
            R = 0
        else:
            loss, R = infonce(scores, device=device)

    elif divergence == "NWJ":
        if "deranged" in architecture:
            loss, R = nwj_deranged(D_value_1, D_value_2, device=device)
        else:
            loss, R = nwj(scores, device=device)

    elif divergence == "SL":
        if "deranged" in architecture:
            loss, R = sl_fdime_deranged(D_value_1, D_value_2, device=device)
        else:
            loss, R = sl_fdime_e(scores, device=device)

    return loss, R

###################### DERANGED ARCHITECTURES ########################

def kl_fdime_deranged(D_value_1, D_value_2, alpha, device="cpu"):
    """KL cost function"""
    eps = 1e-5
    batch_size_1 = D_value_1.size(0)
    batch_size_2 = D_value_2.size(0)
    valid_1 = torch.ones((batch_size_1, 1), device=device)
    valid_2 = torch.ones((batch_size_2, 1), device=device)
    loss_1 = my_binary_crossentropy(valid_1, D_value_1) * alpha
    loss_2 = wasserstein_loss(valid_2, D_value_2)
    loss = loss_1 + loss_2
    R = D_value_1 / alpha
    return loss, R

def gan_fdime_deranged(D_value_1, D_value_2, device="cpu"):
    """GAN cost function"""
    BCE = nn.BCELoss()
    batch_size_1 = D_value_1.size(0)
    batch_size_2 = D_value_2.size(0)
    valid_2 = torch.ones((batch_size_2, 1), device=device)
    fake_1 = torch.zeros((batch_size_1, 1), device=device)
    loss_1 = BCE(D_value_1, fake_1)
    loss_2 = BCE(D_value_2, valid_2)
    loss = loss_1 + loss_2
    R = (1 - D_value_1) / D_value_1
    return loss, R

def hd_fdime_deranged(D_value_1, D_value_2, device="cpu"):
    """HD cost function """
    batch_size_1 = D_value_1.size(0)
    batch_size_2 = D_value_2.size(0)
    valid_1 = torch.ones((batch_size_1, 1), device=device)
    valid_2 = torch.ones((batch_size_2, 1), device=device)
    loss_1 = wasserstein_loss(valid_1, D_value_1)
    loss_2 = reciprocal_loss(valid_2, D_value_2)
    loss = loss_1 + loss_2
    R = 1 / (D_value_1 ** 2)
    return loss, R

def js_fgan_lower_bound_modified(D_value_1, D_value_2):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    return -1 * F.softplus(-1 * D_value_1).mean() - F.softplus(D_value_2).mean()

def smile_deranged(D_value_1, D_value_2, tau, device="cpu"):
    """SMILE cost function """
    eps = 1e-5
    D_value_2_ = torch.clamp(D_value_2, -tau, tau)  # -> il -
    dv = D_value_1.mean() - torch.log(torch.mean(torch.exp(D_value_2_)) + eps)
    js = js_fgan_lower_bound_modified(D_value_1, D_value_2)
    with torch.no_grad():
        dv_js = dv - js
    loss = -(js + dv_js)
    R = torch.exp(js + dv_js)
    return loss, R

def mine_ma_deranged(D_value_1, D_value_2, buffer, momentum=0.9, device="cpu"):
    """Mine cost function using the deranged architecture"""
    if buffer is None:
        buffer = torch.tensor(1.0)

    loss_1 = torch.mean(D_value_1)
    buffer_update = logmeanexp_loss(D_value_2, device=device).exp()
    with torch.no_grad():
        second_term = logmeanexp_loss(D_value_2, device=device)
        buffer_new = buffer * momentum + buffer_update * (1-momentum)
        buffer_new = torch.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new
    third_term_grad = buffer_update / buffer_new
    loss = -(loss_1 - second_term - third_term_grad + third_term_no_grad)
    R = torch.exp(-loss)
    return loss, R, buffer_update  # buffer_new

def rkl_fdime_deranged(D_value_1, D_value_2, device="cpu"):
    """Reverse KL cost function"""
    eps = 1e-5
    loss_1 = torch.mean(torch.pow(D_value_1 + eps, -1))
    loss_2 = torch.mean(torch.log(D_value_2 + eps))
    loss = loss_1 + loss_2
    return loss, D_value_1

def sl_fdime_deranged(D_value_1, D_value_2, device="cpu"):
    eps = 1e-5
    loss_1 = torch.mean(D_value_1)
    loss_2 = torch.mean(torch.log(D_value_2 + eps) - D_value_2)
    R = (1-D_value_1)/D_value_1
    return loss_1-loss_2, R

def tuba_deranged(D_value_1, D_value_2, log_baseline=None):
    """TUBA cost function implemented for the 'deranged-type' architectures"""
    if log_baseline is not None:
        D_value_1 -= log_baseline[:, None]
        D_value_2 -= log_baseline[:, None]
    joint_term = D_value_1.mean()
    marg_term = logmeanexp_loss(D_value_2).exp()
    return -(1. + joint_term - marg_term)

def nwj_deranged(D_value_1, D_value_2, device="cpu"):
    """NWJ cost function"""
    loss = tuba_deranged(D_value_1 - 1., D_value_2 - 1)
    R = torch.exp(-loss)
    return loss, R

#################################### CONCAT - SEPARABLE ARCHITECTURES ###########################################

def logmeanexp_diag(x, device='cpu'):
    """Compute logmeanexp over the diagonal elements of x. The diagonal elements of x contain the mutual information
    over the samples of the joint pdf."""
    batch_size = x.size(0)
    eps = 1e-5
    logsumexp = torch.logsumexp(x.diag(), dim=(0,))
    num_elem = batch_size

    return logsumexp - torch.log(torch.tensor(num_elem).float() + eps).to(device)


def logmeanexp_loss(y_pred, device="cpu"):
    eps = 1e-5
    batch_size = y_pred.size(0)
    logsumexp = torch.logsumexp(y_pred, dim=(0,))
    return logsumexp - torch.log(torch.tensor(batch_size).float() + eps).to(device)

def logmeanexp_nodiag(x, dim=None, device='cpu'):
    """Compute the logmeanexp over the nondiagonal elements, which correspond to the mutual information of the points
    generated from the product of marginals."""
    eps = 1e-5
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)
    # logsumexp of the elements outside the diagonal (subtract -infinity because the exponential of -inf is 0)
    logsumexp = torch.logsumexp(x - torch.diag(np.inf * torch.ones(batch_size).to(device)).to(device), dim=dim)
    try:
        if len(dim) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except ValueError:
        num_elem = batch_size - 1
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)

def tuba(scores, log_baseline=None, device="cpu"):
    """TUBA cost function implemented for the architectures 'joint' and 'separable'"""
    if log_baseline is not None:
        scores -= log_baseline[:, None]
    joint_term = scores.diag().mean()
    marg_term = logmeanexp_nodiag(scores, device=device).exp()
    return -(1. + joint_term - marg_term)

def nwj(scores, device="cpu"):
    """NWJ cost function"""
    loss = tuba(scores - 1., device=device)
    R = torch.exp(-loss)
    return loss, R

def infonce(scores, device="cpu"):
    """INFO_NCE cost function"""
    nll = scores.diag().mean() - scores.logsumexp(dim=1)
    mi = torch.tensor(scores.size(0), device=device).float().log() + nll
    mi = mi.mean()
    R = torch.exp(mi)
    return -mi, R

def kl_fdime_e(scores, device="cpu"):
    """KL cost function"""
    eps = 1e-7
    scores_diag = scores.diag()
    n = scores.size(0)
    scores_no_diag = scores - scores_diag * torch.eye(n, device=device)

    loss_1 = -torch.mean(torch.log(scores_diag + eps))
    loss_2 = torch.sum(scores_no_diag) / (n*(n-1))
    loss = loss_1 + loss_2
    return loss, scores_diag

def gan_fdime_e(scores, device="cpu"):
    """GAN cost function"""
    eps = 1e-5
    batch_size = scores.size(0)
    scores_diag = scores.diag()
    scores_no_diag = scores - scores_diag*torch.eye(batch_size, device=device) + torch.eye(batch_size, device=device)
    R = (1 - scores_diag) / scores_diag
    loss_1 = torch.mean(torch.log(torch.ones(scores_diag.shape, device=device) - scores_diag + eps))
    loss_2 = torch.sum(torch.log(scores_no_diag + eps)) / (batch_size*(batch_size-1))
    return -(loss_1+loss_2), R

def hd_fdime_e(scores, device="cpu"):
    """HD cost function """
    eps = 1e-5
    Eps = 1e7
    scores_diag = scores.diag()
    n = scores.size(0)
    scores_no_diag = scores + Eps * torch.eye(n, device=device)
    loss_1 = torch.mean(scores_diag)
    loss_2 = torch.sum(torch.pow(scores_no_diag, -1))/(n*(n-1))
    loss = -(2 - loss_1 - loss_2)
    return loss, 1 / (scores_diag**2)

def js_fgan_lower_bound(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) - torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term

def smile(f, clip=None, device="cpu"):
    """SMILE cost function"""
    if clip is not None:
        f_ = torch.clamp(f, -clip, clip)
    else:
        f_ = f
    z = logmeanexp_nodiag(f_, dim=(0, 1), device=device)
    dv = f.diag().mean() - z
    js = js_fgan_lower_bound(f)
    with torch.no_grad():
        dv_js = dv - js
    loss = -(js + dv_js)
    R = torch.exp(js + dv_js)
    return loss, R

def mine_ma(f, buffer=None, momentum=0.9, device="cpu"):
    """MINE cost function"""
    buffer = None
    if buffer is None:
        buffer = torch.tensor(1.0)
    first_term = f.diag().mean()

    buffer_update = logmeanexp_nodiag(f, device=device).exp()
    with torch.no_grad():
        second_term = logmeanexp_nodiag(f, device=device)
        buffer_new = buffer * momentum + buffer_update * (1 - momentum)
        buffer_new = torch.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new
    third_term_grad = buffer_update / buffer_new
    loss = -(first_term - second_term - third_term_grad + third_term_no_grad)
    R = torch.exp(-loss)
    return loss, R, buffer_update

def rkl_fdime_e(scores, device="cpu"):
    """Reverse KL cost function"""
    eps = 1e-5
    n = scores.size(0)
    scores_diag = scores.diag()
    scores_no_diag = scores - scores_diag * torch.eye(n, device=device) + torch.eye(n, device=device)
    loss_1 = torch.mean(torch.pow(scores_diag + eps, -1))
    loss_2 = torch.sum(torch.log(scores_no_diag + eps))/(n*(n-1))
    loss = loss_1 + loss_2
    return loss, scores_diag

def sl_fdime_e(scores, device="cpu"):
    eps = 1e-7
    n = scores.size(0)
    scores = scores + eps
    scores_diag = scores.diag()
    scores_no_diag = scores - scores_diag * torch.eye(n, device=device) + torch.eye(n, device=device)
    loss_1 = torch.mean(scores_diag)
    loss_2 = torch.sum(torch.log(scores_no_diag) - (scores_no_diag-torch.eye(n, device=device)))/(n*(n-1))
    R = (1-scores_diag)/scores_diag
    return loss_1-loss_2, R

#####################################################################################################

def plot_staircases(staircases, proc_params, opt_params, latent_dim):
    architecture_2_color = {
        'joint': '#1f77b4',
        'separable': '#ff7f0e',
        'deranged': '#2ca02c',
    }
    n_divergences = len(proc_params["divergences"])
    n_architectures = len(proc_params['architectures'])
    fig, sbplts = plt.subplots(len(proc_params["modes"]), n_divergences, figsize=(4 * n_divergences, 4 * len(proc_params["modes"])))
    len_step = proc_params['len_step']
    tot_len_stairs = proc_params['tot_len_stairs']
    for idx, mode in enumerate(proc_params["modes"]):
        mode_sbplt = sbplts[idx]
        i = 0
        if n_divergences > 1:
            for divergence in proc_params['divergences']:
                mode_sbplt[i].plot(range(tot_len_stairs), np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                            linewidth=1, c='k', linestyle="dashed")
                for architecture in proc_params['architectures']:
                    if divergence == "CPC" and "deranged" in architecture:
                        pass
                    else:
                        fDIME_training_staircase_smooth = pd.Series(staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(span=200).mean()
                        sm = mode_sbplt[i].plot(range(tot_len_stairs), staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                                 linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                        mode_sbplt[i].plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture, linewidth=1, c=sm.get_color())
                mode_sbplt[i].plot(range(tot_len_stairs), np.repeat(proc_params['levels_MI'], len_step), label="True MI", linewidth=1, c='k')
                if i==0:
                    mode_sbplt[i].set_ylabel('MI [nats]', fontsize=18)
                if divergence=="GAN" or divergence=="NWJ":
                    mode_sbplt[i].legend(loc="best", fontsize=10)
                mode_sbplt[i].set_xlabel('Steps', fontsize=18)
                if divergence in ["RKL", "SL", "GAN", "KL", "HD"]:
                    mode_sbplt[i].set_title("{}-DIME".format(divergence), fontsize=20)
                elif divergence in ["NWJ"]:
                    mode_sbplt[i].set_title("NWJ-{}".format(mode), fontsize=20)
                else:
                    mode_sbplt[i].set_title(divergence, fontsize=20)
                mode_sbplt[i].set_xlim([0, tot_len_stairs])
                mode_sbplt[i].set_ylim([0, proc_params['levels_MI'][-1]+2])
                i += 1
        else:
            divergence = proc_params['divergences'][0]
            if divergence == "CPC":
                mode_sbplt.plot(range(tot_len_stairs),
                                    np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                                    linewidth=1, c='k', linestyle="dashed")
            for architecture in proc_params['architectures']:
                if divergence == "CPC" and "deranged" in architecture:
                    pass
                else:
                    fDIME_training_staircase_smooth = pd.Series(
                        staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(
                        span=200).mean()
                    sm = mode_sbplt.plot(range(tot_len_stairs), staircases[
                        f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                             linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                    mode_sbplt.plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture,
                                        linewidth=1, c=sm.get_color())
            mode_sbplt.plot(range(tot_len_stairs), np.repeat(proc_params['levels_MI'], len_step), label="True MI",
                                linewidth=1, c='k')
            mode_sbplt.set_ylabel('MI [nats]', fontsize=18)
            mode_sbplt.legend(loc="best")
            mode_sbplt.set_xlabel('Steps', fontsize=18)
            if divergence in ["RKL", "SL", "GAN", "KL", "HD"]:
                mode_sbplt.set_title("{}-DIME".format(divergence), fontsize=20)
            elif divergence in ["NWJ"]:
                mode_sbplt.set_title("NWJ-{}".format(mode), fontsize=20)
            else:
                mode_sbplt.set_title(divergence, fontsize=20)
            mode_sbplt.set_xlim([0, tot_len_stairs])
            mode_sbplt.set_ylim([0, proc_params['levels_MI'][-1] + 2])

    plt.gcf().tight_layout()
    plt.savefig("Results/Stairs/allStaircases_d{}_bs{}_arc{}.svg".format(latent_dim, opt_params['batch_size'], proc_params["architectures"][0]))


def compute_MI_given_eps_unif(eps):
    if eps > 0.5:
        return 1/(4*eps)
    else:
        return eps - np.log(2*eps)


def plot_staircases_unif(staircases, proc_params, opt_params, latent_dim):
    architecture_2_color = {
        'joint': '#1f77b4',
        'separable': '#ff7f0e',
        'deranged': '#2ca02c'
    }
    n_divergences = len(proc_params["divergences"])
    n_architectures = len(proc_params['architectures'])
    fig, mode_sbplt = plt.subplots(len(proc_params["modes"]), n_divergences, figsize=(4 * n_divergences, 4 * len(proc_params["modes"])))
    len_step = proc_params['len_step']
    tot_len_stairs = proc_params['tot_len_stairs']
    mode = "uniform"
    i = 0
    if n_divergences > 1:
        for divergence in proc_params['divergences']:
            if divergence == "CPC":
                mode_sbplt[i].plot(range(tot_len_stairs), np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                         linewidth=1, c='k', linestyle="dashed")
            for architecture in proc_params['architectures']:
                if divergence == "CPC" and "deranged" in architecture:
                    pass
                else:
                    fDIME_training_staircase_smooth = pd.Series(staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(span=200).mean()
                    sm = mode_sbplt[i].plot(range(tot_len_stairs), staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                             linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                    mode_sbplt[i].plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture, linewidth=1, c=sm.get_color())
            true_MIs = [compute_MI_given_eps_unif(eps) for eps in proc_params['levels_eps']]
            mode_sbplt[i].plot(range(tot_len_stairs), np.repeat(true_MIs, len_step), label="True MI", linewidth=1, c='k')
            if i==0:
                mode_sbplt[i].set_ylabel('MI [nats]', fontsize=18)
            if divergence=="GAN" or divergence=="NWJ":
                mode_sbplt[i].legend(loc="best", fontsize=10)
            mode_sbplt[i].set_xlabel('Steps', fontsize=18)
            if divergence in ["RKL", "SL", "GAN", "KL", "HD"]:
                mode_sbplt[i].set_title("{}-DIME".format(divergence), fontsize=20)
            elif divergence in ["NWJ"]:
                mode_sbplt[i].set_title("NWJ-{}".format(mode), fontsize=20)
            else:
                mode_sbplt[i].set_title(divergence, fontsize=20)
            mode_sbplt[i].set_xlim([0, tot_len_stairs])
            mode_sbplt[i].set_ylim([0, 3])
            i += 1
    else:
        divergence = proc_params['divergences'][0]
        if divergence == "CPC":
            mode_sbplt.plot(range(tot_len_stairs),
                                np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                                linewidth=1, c='k', linestyle="dashed")
        for architecture in proc_params['architectures']:
            if divergence == "CPC" and "deranged" in architecture:
                pass
            else:
                fDIME_training_staircase_smooth = pd.Series(
                    staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(
                    span=200).mean()
                sm = mode_sbplt.plot(range(tot_len_stairs), staircases[
                    f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                         linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                mode_sbplt.plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture,
                                    linewidth=1, c=sm.get_color())
        true_MIs = [compute_MI_given_eps_unif(eps) for eps in proc_params['levels_eps']]
        mode_sbplt.plot(range(tot_len_stairs), np.repeat(true_MIs, len_step), label="True MI", linewidth=1, c='k')
        mode_sbplt.set_ylabel('MI [nats]', fontsize=18)
        mode_sbplt.legend(loc="best")
        mode_sbplt.set_xlabel('Steps', fontsize=18)
        if divergence in ["RKL", "SL", "GAN", "KL", "HD"]:
            mode_sbplt.set_title("{}-DIME".format(divergence), fontsize=20)
        elif divergence in ["NWJ"]:
            mode_sbplt.set_title("NWJ-{}".format(mode), fontsize=20)
        else:
            mode_sbplt.set_title(divergence, fontsize=20)
        mode_sbplt.set_xlim([0, tot_len_stairs])
        mode_sbplt.set_ylim([0, 3])

    plt.gcf().tight_layout()
    plt.savefig("Results/Stairs/allStaircases_d{}_bs{}_arc{}_scenuniform.svg".format(latent_dim, opt_params['batch_size'], proc_params["architectures"][0]))


def plot_staircases_swiss(staircases, proc_params, opt_params, latent_dim):
    architecture_2_color = {
        'joint': '#1f77b4',
        'separable': '#ff7f0e',
        'deranged': '#2ca02c'
    }
    n_divergences = len(proc_params["divergences"])
    n_architectures = len(proc_params['architectures'])
    fig, mode_sbplt = plt.subplots(len(proc_params["modes"]), n_divergences, figsize=(4 * n_divergences, 4 * len(proc_params["modes"])))
    len_step = proc_params['len_step']
    tot_len_stairs = proc_params['tot_len_stairs']
    mode = "swiss"
    i = 0
    if n_divergences > 1:
        for divergence in proc_params['divergences']:
            if divergence == "CPC":
                mode_sbplt[i].plot(range(tot_len_stairs), np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                         linewidth=1, c='k', linestyle="dashed")
            for architecture in proc_params['architectures']:
                if divergence == "CPC" and "deranged" in architecture:
                    pass
                else:
                    fDIME_training_staircase_smooth = pd.Series(staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(span=200).mean()
                    sm = mode_sbplt[i].plot(range(tot_len_stairs), staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                             linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                    mode_sbplt[i].plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture, linewidth=1, c=sm.get_color())
            mode_sbplt[i].plot(range(tot_len_stairs), np.repeat(proc_params['levels_MI'], len_step), label="True MI", linewidth=1, c='k')
            if i==0:
                mode_sbplt[i].set_ylabel('MI [nats]', fontsize=18)
            if divergence=="GAN" or divergence=="NWJ":
                mode_sbplt[i].legend(loc="best", fontsize=10)
            mode_sbplt[i].set_xlabel('Steps', fontsize=18)
            if divergence in ["RKL", "SL", "GAN", "KL", "HD"]:
                mode_sbplt[i].set_title("{}-DIME".format(divergence), fontsize=20)
            elif divergence in ["NWJ"]:
                mode_sbplt[i].set_title("NWJ-{}".format(mode), fontsize=20)
            else:
                mode_sbplt[i].set_title(divergence, fontsize=20)
            mode_sbplt[i].set_xlim([0, tot_len_stairs])
            mode_sbplt[i].set_ylim([0, proc_params['levels_MI'][-1]+2])
            i += 1
    else:
        divergence = proc_params['divergences'][0]
        if divergence == "CPC":
            mode_sbplt.plot(range(tot_len_stairs),
                                np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                                linewidth=1, c='k', linestyle="dashed")
        for architecture in proc_params['architectures']:
            if divergence == "CPC" and "deranged" in architecture:
                pass
            else:
                fDIME_training_staircase_smooth = pd.Series(
                    staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(
                    span=200).mean()
                sm = mode_sbplt.plot(range(tot_len_stairs), staircases[
                    f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                         linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                mode_sbplt.plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture,
                                    linewidth=1, c=sm.get_color())
        mode_sbplt.plot(range(tot_len_stairs), np.repeat(proc_params['levels_MI'], len_step), label="True MI",
                            linewidth=1, c='k')
        mode_sbplt.set_ylabel('MI [nats]', fontsize=18)
        mode_sbplt.legend(loc="best")
        mode_sbplt.set_xlabel('Steps', fontsize=18)
        if divergence in ["RKL", "SL", "GAN", "KL", "HD"]:
            mode_sbplt.set_title("{}-DIME".format(divergence), fontsize=20)
        elif divergence in ["NWJ"]:
            mode_sbplt.set_title("NWJ-{}".format(mode), fontsize=20)
        else:
            mode_sbplt.set_title(divergence, fontsize=20)
        mode_sbplt.set_xlim([0, tot_len_stairs])
        mode_sbplt.set_ylim([0, proc_params['levels_MI'][-1] + 2])

    plt.gcf().tight_layout()
    plt.savefig("Results/Stairs/allStaircases_d{}_bs{}_arc{}_scenswiss.svg".format(latent_dim, opt_params['batch_size'], proc_params["architectures"][0]))


def _differential_entropy(k, dof):
    """
    Differential entropy of a :math:`Student-t(0, I_k, dof)`.
    """
    half_sum = 0.5 * (dof + k)
    digamma_term = half_sum * (scipy.special.digamma(half_sum) - scipy.special.digamma(0.5 * dof))
    log_term = -np.log(scipy.special.gamma(half_sum)) + np.log(scipy.special.gamma(0.5 * dof)) + 0.5 * k * np.log(dof * np.pi)
    return log_term + digamma_term


def compute_MI_given_df_stud(df, d):
    rho = 0
    I_Xt_Yt = -d/2 * np.log(1 - rho**2)
    h_x = _differential_entropy(k=d, dof=df)
    h_y = _differential_entropy(k=d, dof=df)
    h_xy = _differential_entropy(k=2*d, dof=df)
    c = h_x + h_y - h_xy
    return I_Xt_Yt + c

def plot_staircases_student(staircases, proc_params, opt_params, latent_dim):
    architecture_2_color = {
        'joint': '#1f77b4',
        'separable': '#ff7f0e',
        'deranged': '#2ca02c'
    }
    mode = "student"
    n_divergences = len(proc_params["divergences"])
    n_architectures = len(proc_params['architectures'])
    fig, mode_sbplt = plt.subplots(len(proc_params["modes"]), n_divergences,
                               figsize=(4 * n_divergences, 4 * len(proc_params["modes"])))
    len_step = proc_params['len_step']
    tot_len_stairs = proc_params['tot_len_stairs']
    i = 0
    if n_divergences > 1:
        for divergence in proc_params['divergences']:
            if divergence == "CPC":
                mode_sbplt[i].plot(range(tot_len_stairs),
                                   np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)),
                                   label="ln(bs)",
                                   linewidth=1, c='k', linestyle="dashed")
            for architecture in proc_params['architectures']:
                if divergence == "CPC" and "deranged" in architecture:
                    pass
                else:
                    fDIME_training_staircase_smooth = pd.Series(
                        staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(
                        span=200).mean()
                    sm = mode_sbplt[i].plot(range(tot_len_stairs), staircases[
                        f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                            linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                    mode_sbplt[i].plot(range(tot_len_stairs), fDIME_training_staircase_smooth,
                                       label=architecture, linewidth=1, c=sm.get_color())
            print("compute_MI_given_df_stud(df, latent_dim): ", compute_MI_given_df_stud(proc_params["levels_df"][0], latent_dim))
            true_MIs = [compute_MI_given_df_stud(df, latent_dim) for df in proc_params["levels_df"]]
            mode_sbplt[i].plot(range(tot_len_stairs), np.repeat(true_MIs, len_step), label="True MI",
                               linewidth=1, c='k')
            if i == 0:
                mode_sbplt[i].set_ylabel('MI [nats]', fontsize=18)
            if divergence == "GAN" or divergence == "NWJ":
                mode_sbplt[i].legend(loc="best", fontsize=10)
            mode_sbplt[i].set_xlabel('Steps', fontsize=18)
            if divergence in ["RKL", "SL", "GAN", "KL", "HD"]:
                mode_sbplt[i].set_title("{}-DIME".format(divergence), fontsize=20)
            elif divergence in ["NWJ"]:
                mode_sbplt[i].set_title("NWJ-{}".format(mode), fontsize=20)
            else:
                mode_sbplt[i].set_title(divergence, fontsize=20)
            mode_sbplt[i].set_xlim([0, tot_len_stairs])
            mode_sbplt[i].set_ylim([0, proc_params['levels_df'][-1] + 2])
            i += 1
    else:
        divergence = proc_params['divergences'][0]
        if divergence == "CPC":
            mode_sbplt.plot(range(tot_len_stairs),
                            np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                            linewidth=1, c='k', linestyle="dashed")
        for architecture in proc_params['architectures']:
            if divergence == "CPC" and "deranged" in architecture:
                pass
            else:
                fDIME_training_staircase_smooth = pd.Series(
                    staircases[f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(
                    span=200).mean()
                sm = mode_sbplt.plot(range(tot_len_stairs), staircases[
                    f'{mode}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                     linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                mode_sbplt.plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture,
                                linewidth=1, c=sm.get_color())
        print("compute_MI_given_df_stud(df, latent_dim): ", compute_MI_given_df_stud(proc_params["levels_df"][0], latent_dim))
        true_MIs = [compute_MI_given_df_stud(df, latent_dim) for df in proc_params['levels_df']]
        mode_sbplt.plot(range(tot_len_stairs), np.repeat(true_MIs, len_step), label="True MI", linewidth=1,
                        c='k')
        mode_sbplt.set_ylabel('MI [nats]', fontsize=18)
        mode_sbplt.legend(loc="best")
        mode_sbplt.set_xlabel('Steps', fontsize=18)
        if divergence in ["RKL", "SL", "GAN", "KL", "HD"]:
            mode_sbplt.set_title("{}-DIME".format(divergence), fontsize=20)
        elif divergence in ["NWJ"]:
            mode_sbplt.set_title("NWJ-{}".format(mode), fontsize=20)
        else:
            mode_sbplt.set_title(divergence, fontsize=20)
        mode_sbplt.set_xlim([0, tot_len_stairs])
        mode_sbplt.set_ylim([0, proc_params['levels_df'][-1] + 2])

    plt.gcf().tight_layout()
    plt.savefig("Results/Stairs/allStaircases_d{}_bs{}_arc{}_scenstudent.svg".format(latent_dim, opt_params['batch_size'], proc_params["architectures"][0]))


def save_time_dict(time_dict, latent_dim, batch_size, proc_params, scenario):
    with open("Results/Stairs/time_dictionary_d{}_bs{}_arc{}_scen{}.json".format(latent_dim, batch_size, proc_params["architectures"][0], scenario), "w") as fp:
        json.dump(time_dict.copy(), fp)


def save_dict_lists_csv(path, dictionary):
    with open(path, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dictionary.keys())
        writer.writerows(zip(*dictionary.values()))


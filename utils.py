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
from torchvision.utils import save_image
import cv2
import json
import math
import time
import numpy as np
import scipy.io as sio
import csv


# Some functions in the code are adapted from: https://github.com/ermongroup/smile-mi-estimator  (04/2023)

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128, cubic=None, device="cpu"):
    """Generate samples from a correlated Gaussian distribution with correlation rho."""
    x, eps = torch.chunk(torch.randn(batch_size, 2 * dim), 2, dim=1)
    y = rho * x + torch.sqrt(torch.tensor(1. - rho**2).float()) * eps
    if cubic:
        y = y ** 3
    return x.to(device), y.to(device)


def sample_gaussian(batch_size, latent_dim, eps, cubic=False):
    """Generate samples from a Gaussian distribution of the type Y = X + N"""
    x = np.random.normal(0, 1, (batch_size, latent_dim))
    y = x + eps * np.random.normal(0, 1, (batch_size, latent_dim))
    if cubic:
        y = y**3
    return x, y


def mi_to_rho(dim, mi):
    """Obtain the rho for Gaussian, given the true mutual information."""
    return np.sqrt(1 - np.exp(-2.0 / dim * mi))


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
    dismutations = True
    data_xy = torch.hstack((data_x, data_y))
    if dismutations:  # Derangement
        data_y_shuffle = torch.index_select(data_y, 0, derangement(list(range(data_y.shape[0])), device))
        # ordered_derangement = [(idx + 1) % data_y.shape[0] for idx in range(data_y.shape[0])]
        # data_y_shuffle = np.take(data_y, ordered_derangement, axis=0, out=data_y)
    else:  # Permutation
        data_y_shuffle = torch.index_select(data_y, 0, torch.Tensor(np.random.permutation(data_y.shape[0])).int().to(device))

    data_x_y = torch.hstack((data_x, data_y_shuffle))
    return data_xy, data_x_y


def wasserstein_loss(y_true, y_pred):
    """Computes Wasserstein loss"""
    return torch.mean(y_true * y_pred)


def reciprocal_loss(y_true, y_pred):
    """Computes reciprocal loss"""
    return torch.mean(torch.pow(y_true*y_pred, -1))


def my_binary_crossentropy(y_true, y_pred):
    """Custom binary cross-entropy"""
    eps = 1e-7
    return -torch.mean(torch.log(y_true)+torch.log(y_pred + eps))


def logmeanexp_loss(y_pred, device="cpu"):
    """Computes the log of the mean of the exp."""
    eps = 1e-5
    batch_size = y_pred.size(0)
    logsumexp = torch.logsumexp(y_pred, dim=(0,))
    return logsumexp - torch.log(torch.tensor(batch_size).float() + eps).to(device)


def save_cons_test_results(fdime_test_dict, proc_params, tt, help_dict, digits):
    """Save the MI estimates"""
    for architecture in proc_params['architectures']:
        for divergence in proc_params['divergences']:
            tmp_list_i = []
            for it_var in proc_params[help_dict[tt[3:]]]:
                key_tmp = "{}_{}_{}_{}".format(divergence, architecture, it_var, tt)
                tmp_list_i.append(fdime_test_dict[key_tmp].item())
            print("going to save the estimate I(x;y)")
            with open("Results/MI_estimates/mi_{}_{}_{}_{}".format(architecture, divergence, tt, digits), "w") as f:
                json.dump(tmp_list_i, f)


def plot_staircase(training_staircase, tot_len_stairs, divergence, len_step, batch_size, label, saving_name, levels_MI=5):
    """Plot the staircase with the mutual information ground truth and the estimate output of the network."""
    fig = plt.figure()
    fDIME_training_staircase_smooth = pd.Series(training_staircase).ewm(span=200).mean()
    sm = plt.plot(range(tot_len_stairs), training_staircase,
                  linewidth=1, alpha=0.5)[0]
    plt.plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label="{}-DIME".format(divergence), linewidth=1,
             c=sm.get_color())

    if "Loss" not in label:
        plt.plot(range(tot_len_stairs), np.repeat(levels_MI, len_step), label="True MI", linewidth=1, c='k')
        plt.plot(range(tot_len_stairs), np.log(batch_size) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                 linewidth=1, c='k', linestyle="dashed")
    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel(label)
    if label != "R" and label != "Loss":
        plt.ylim(0, levels_MI[-1]+2)
    plt.xlim(0, tot_len_stairs)
    plt.title(divergence)
    plt.savefig(saving_name)
    plt.grid()
    plt.clf()


def plot_staircases(staircases, proc_params, opt_params, latent_dim):
    """Plot all the MI estimates for all the estimators"""
    architecture_2_color = {
        'joint': '#1f77b4',
        'separable': '#ff7f0e',
        'deranged': '#2ca02c',
        'ad_hoc': '#9467bd'
    }
    n_divergences = len(proc_params['divergences'])
    fig, sbplts = plt.subplots(2, n_divergences, figsize=(4 * n_divergences, 4 * 2))
    len_step = proc_params['len_step']
    tot_len_stairs = proc_params['tot_len_stairs']
    for idx, cubic in enumerate([False, True]):
        cubic_sbplt = sbplts[idx]
        i = 0
        if n_divergences > 1:
            for divergence in proc_params['divergences']:
                if divergence=="CPC":
                    cubic_sbplt[i].plot(range(tot_len_stairs), np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                             linewidth=1, c='k', linestyle="dashed")
                if divergence == "NJEE":
                    architecture = "ad_hoc"
                    fDIME_training_staircase_smooth = pd.Series(staircases[f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(span=200).mean()
                    sm = cubic_sbplt[i].plot(range(tot_len_stairs), staircases[f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                             linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                    cubic_sbplt[i].plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture,
                                        linewidth=1, c=sm.get_color())
                else:
                    for architecture in proc_params['architectures']:
                        if divergence == "CPC" and architecture == "deranged":
                            pass
                        else:
                            fDIME_training_staircase_smooth = pd.Series(staircases[f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(span=200).mean()
                            sm = cubic_sbplt[i].plot(range(tot_len_stairs), staircases[f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                                     linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                            cubic_sbplt[i].plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture, linewidth=1, c=sm.get_color())
                cubic_sbplt[i].plot(range(tot_len_stairs), np.repeat(proc_params['levels_MI'], len_step), label="True MI", linewidth=1, c='k')
                if i == 0:
                    cubic_sbplt[i].set_ylabel('MI [nats]', fontsize=18)
                if divergence == "NJEE" or divergence == "GAN" or divergence == "NWJ":
                    cubic_sbplt[i].legend(loc="best", fontsize=12)
                cubic_sbplt[i].set_xlabel('Steps', fontsize=18)
                if divergence in ["GAN", "KL", "HD"]:
                    cubic_sbplt[i].set_title("{}-DIME".format(divergence), fontsize=20)
                else:
                    cubic_sbplt[i].set_title(divergence, fontsize=20)
                cubic_sbplt[i].set_xlim([0, tot_len_stairs])
                cubic_sbplt[i].set_ylim([0, proc_params['levels_MI'][-1]+2])
                i += 1
        else:
            divergence = proc_params['divergences'][0]
            if divergence == "CPC":
                cubic_sbplt.plot(range(tot_len_stairs),
                                    np.log(opt_params['batch_size']) * np.ones((tot_len_stairs, 1)), label="ln(bs)",
                                    linewidth=1, c='k', linestyle="dashed")
            if divergence == "NJEE":
                architecture = "ad_hoc"
                fDIME_training_staircase_smooth = pd.Series(
                    staircases[f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(span=200).mean()
                sm = cubic_sbplt.plot(range(tot_len_stairs),
                                         staircases[f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                         linewidth=1, alpha=0.2, c=architecture_2_color[architecture])[0]
                cubic_sbplt.plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture,
                                    linewidth=1, c=sm.get_color())
            else:
                for architecture in proc_params['architectures']:
                    if divergence == "CPC" and architecture == "deranged":
                        pass
                    else:
                        fDIME_training_staircase_smooth = pd.Series(
                            staircases[f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}']).ewm(
                            span=200).mean()
                        sm = cubic_sbplt.plot(range(tot_len_stairs), staircases[
                            f'{cubic}_{divergence}_{architecture}_{opt_params["batch_size"]}'],
                                                 linewidth=1, alpha=0.3, c=architecture_2_color[architecture])[0]
                        cubic_sbplt.plot(range(tot_len_stairs), fDIME_training_staircase_smooth, label=architecture,
                                            linewidth=1, c=sm.get_color())
            cubic_sbplt.plot(range(tot_len_stairs), np.repeat(proc_params['levels_MI'], len_step), label="True MI",
                                linewidth=1, c='k')
            cubic_sbplt.set_ylabel('MI [nats]', fontsize=18)
            cubic_sbplt.legend(loc="best")
            cubic_sbplt.set_xlabel('Steps', fontsize=18)
            if divergence in ["GAN", "KL", "HD"]:
                cubic_sbplt.set_title("{}-DIME".format(divergence), fontsize=20)
            else:
                cubic_sbplt.set_title(divergence, fontsize=20)
            cubic_sbplt.set_xlim([0, tot_len_stairs])
            cubic_sbplt.set_ylim([0, proc_params['levels_MI'][-1] + 2])

    plt.gcf().tight_layout()
    plt.savefig("Results/Stairs/allStaircases_d{}_bs{}.svg".format(latent_dim, opt_params['batch_size']))


def save_dict_lists_csv(path, dictionary):
    """Save dictionary into csv file"""
    with open(path, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dictionary.keys())
        writer.writerows(zip(*dictionary.values()))


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


def logmeanexp_diag(x, device='cpu'):
    """Compute logmeanexp over the diagonal elements of x, containing the mutual information
    over the samples of the joint pdf."""
    batch_size = x.size(0)
    eps = 1e-5
    logsumexp = torch.logsumexp(x.diag(), dim=(0,))

    return logsumexp - torch.log(torch.tensor(batch_size).float() + eps).to(device)


def logmeanexp_nodiag(x, dim=None, device='cpu'):
    """Compute the logmeanexp over the nondiagonal elements, corresponding to the mutual information of the points
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
    return logsumexp - torch.log(torch.tensor(num_elem) + eps).to(device)


def tuba_deranged(D_value_1, D_value_2, log_baseline=None):
    """TUBA cost function implemented for the 'deranged-type' architectures"""
    if log_baseline is not None:
        D_value_1 -= log_baseline[:, None]
        D_value_2 -= log_baseline[:, None]
    joint_term = D_value_1.mean()
    marg_term = logmeanexp_loss(D_value_2).exp()
    return -(1. + joint_term - marg_term)


def tuba(scores, log_baseline=None, device="cpu"):
    """TUBA cost function implemented for the architectures 'joint' and 'separable'"""
    if log_baseline is not None:
        scores -= log_baseline[:, None]
    joint_term = scores.diag().mean()
    marg_term = logmeanexp_nodiag(scores, device=device).exp()
    return -(1. + joint_term - marg_term)


def nwj_deranged(D_value_1, D_value_2, device="cpu"):
    """NWJ cost function for the deranged architecture"""
    loss = tuba_deranged(D_value_1 - 1., D_value_2 - 1)
    R = torch.exp(-loss)
    return loss, R


def nwj(scores, device="cpu"):
    """NWJ cost function for joint and separable architectures"""
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


def kl_fdime_deranged(D_value_1, D_value_2, alpha, device="cpu"):
    """KL cost function for the deranged architecture"""
    eps = 1e-5
    batch_size_1 = D_value_1.size(0)
    batch_size_2 = D_value_2.size(0)
    valid_1 = torch.ones((batch_size_1, 1), device=device)
    valid_2 = torch.ones((batch_size_2, 1), device=device)
    loss_1 = my_binary_crossentropy(valid_1, D_value_1) * alpha
    loss_2 = wasserstein_loss(valid_2, D_value_2)
    loss = loss_1 + loss_2
    J_e = alpha * torch.mean(torch.log(D_value_1 + eps)) - torch.mean(D_value_2)
    VLB_e = J_e / alpha + 1 - np.log(alpha)
    R = D_value_1 / alpha
    return loss, R, VLB_e


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


def gan_fdime_deranged(D_value_1, D_value_2, device="cpu"):
    """GAN cost function for the deranged architecture"""
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


def gan_fdime_e(scores, device="cpu"):
    """GAN cost function"""
    eps = 1e-5
    batch_size = scores.size(0)
    scores_diag = scores.diag()
    scores_no_diag = scores - scores_diag*torch.eye(batch_size, device=device) + torch.eye(batch_size, device=device)
    R = (1 - scores_diag) / scores_diag
    loss_1 = torch.mean(torch.log(torch.ones(scores_diag.shape, device=device) - scores_diag + eps)) #-torch.mean(torch.log(scores_diag + eps)) #
    loss_2 = torch.sum(torch.log(scores_no_diag + eps)) / (batch_size*(batch_size-1))
    return -(loss_1+loss_2), R


def hd_fdime_deranged(D_value_1, D_value_2, device="cpu"):
    """HD cost function for the deranged architecture"""
    batch_size_1 = D_value_1.size(0)
    batch_size_2 = D_value_2.size(0)
    valid_1 = torch.ones((batch_size_1, 1), device=device)
    valid_2 = torch.ones((batch_size_2, 1), device=device)
    loss_1 = wasserstein_loss(valid_1, D_value_1)
    loss_2 = reciprocal_loss(valid_2, D_value_2)
    loss = loss_1 + loss_2
    R = 1 / (D_value_1 ** 2)
    return loss, R


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


def js_fgan_lower_bound_deranged(D_value_1, D_value_2):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016), for the deranged architecture."""
    return -1 * F.softplus(-1 * D_value_1).mean() - F.softplus(D_value_2).mean()


def js_fgan_lower_bound(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) - torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term


def smile_deranged(D_value_1, D_value_2, tau, device="cpu"):
    """SMILE cost function for the deranged architecture"""
    eps = 1e-5
    D_value_2_ = torch.clamp(D_value_2, -tau, tau)
    dv = D_value_1.mean() - torch.log(torch.mean(torch.exp(D_value_2_)) + eps)
    js = js_fgan_lower_bound_deranged(D_value_1, D_value_2)
    with torch.no_grad():
        dv_js = dv - js
    loss = -(js + dv_js)
    R = torch.exp(js + dv_js)
    #loss = -dv
    #R = torch.exp(dv)
    return loss, R


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
    #loss = -dv
    #R = torch.exp(dv)
    return loss, R


def mine_deranged(D_value_1, D_value_2, device="cpu"):
    """MINE cost function for the deranged architecture"""
    loss_1 = - torch.mean(D_value_1)
    loss_2 = logmeanexp_loss(D_value_2, device=device)
    loss = loss_1 + loss_2
    R = torch.exp(-loss)
    return loss, R


def mine_m_deranged(D_value_1, D_value_2, buffer, momentum=0.9, device="cpu"):
    """Mine cost function using the deranged architecture"""
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
    return loss, R, buffer_update


def mine_m(f, buffer=None, momentum=0.9, device="cpu"):
    """MINE cost function for the deranged architecture"""
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


def print_training_title(divergence, architecture):
    """Print the training title"""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("DIVERGENCE-ARCHITECTURE: {}-{}, at time: {}".format(divergence, architecture, current_time))


def load_mnist(digits=False):
    """Load the MNIST dataset"""
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    if digits:
        train_dataset = torchvision.datasets.MNIST('mnist_dataset_training', download=True, train=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST('mnist_dataset_test', download=True, train=False, transform=transform)
    else:
        train_dataset = torchvision.datasets.FashionMNIST('mnist_dataset_training', download=True, train=True,
                                                   transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST('mnist_dataset_test', download=True, train=False, transform=transform)
    # Obtain the dataloader object and return also the number of features
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)
    num_features = images.shape[2] * images.shape[3]
    return train_dataset, test_dataset, num_features


def obtain_y_consistency(x, test_type, it_var, device="cpu"):
    """Obtain batch y given batch x, for the consistency tests. The test type is a combination of two strings:
    the first one identifies the method of consistency test: {"bs", "dp", "ad"}. (baseline, data-processing, additivity)
    The second one identifies the transformation of y: {"mask", "rot", "scale", "tran"}. it_var is the value of the
    iterable variable."""
    y = x.clone()  # x and y are batches
    y = transform_y(y, test_type, it_var)
    #save_image(x[0][0], 'x_0_0.png')
    #save_image(y[0][0], 'y_0_0.png')

    if "bs" in test_type:
        return x, y
    elif "dp" in test_type:
        h_y = y.clone()
        if "mask" in test_type:
            h_y = transform_y(h_y, test_type, it_var=it_var-3)
        else:
            h_y = transform_y(h_y, test_type, it_var=it_var/4)
        x = torch.cat((x, x), dim=1)
        y = torch.cat((y, h_y), dim=1)
        return x, y
    elif "ad" in test_type:
        tmp_x_1 = x.clone()
        tmp_x_2 = x.clone()
        tmp_y_1 = y.clone()
        tmp_y_2 = y.clone()
        shuffled_indices = derangement(list(range(tmp_x_2.shape[0])), device)
        tmp_x_2_shuffled = torch.index_select(tmp_x_2, 0, shuffled_indices)
        tmp_y_2_shuffled = torch.index_select(tmp_y_2, 0, shuffled_indices)
        x = torch.cat((tmp_x_1, tmp_x_2_shuffled), dim=1)
        y = torch.cat((tmp_y_1, tmp_y_2_shuffled), dim=1)
        return x, y


def transform_y(y, test_type, it_var):
    """Apply the transformation based on the test_type"""
    if "mask" in test_type:
        for image in y:
            image = image.squeeze()
            image[it_var:y.shape[2], :] = -1.
    elif "rot" in test_type:
        y = transforms.functional.affine(y, angle=it_var, translate=[0, 0], scale=1, shear=0)
    elif "scale" in test_type:
        y = transforms.functional.affine(y, angle=0, translate=[0, 0], scale=1+it_var, shear=0)
    elif "tran" in test_type:
        y = transforms.functional.affine(y, angle=0, translate=[math.ceil(it_var), math.ceil(it_var)], scale=1, shear=0)
    return y


def compute_loss_ratio(divergence, architecture, device, D_value_1=None, D_value_2=None, scores=None, buffer=None, alpha=1):
    """Compute the value of the loss and R given a certain cost function, a specific neural network and the output of
    such a neural network. R is e^{\hat{I}(X;Y)}."""
    if divergence == 'KL':
        if "deranged" in architecture:
            loss, R, VLB_e = kl_fdime_deranged(D_value_1, D_value_2, alpha=alpha, device=device)
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

    elif divergence == 'MINE':
        if "deranged" in architecture:
            loss, R, buffer = mine_m_deranged(D_value_1, D_value_2, buffer)
        else:
            loss, R, buffer = mine_m(scores, buffer, momentum=0.9, device=device)

    elif divergence == 'SMILE':
        tau = 1.0  # np.inf
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

    return loss, R


def save_time_dict(time_dict, latent_dim, batch_size):
    with open("Results/Stairs/time_dictionary_d{}_bs{}".format(latent_dim, batch_size), "w") as fp:
        json.dump(time_dict, fp)


##################
###### NJEE ######

# Adapted from: https://github.com/YuvalShalev/NJEE  (04/2023)
def making_cov(rho, dims):
    """Creates covariance matrix for correlated Gaussians."""
    cov = np.zeros((2*dims, 2*dims))
    for i in range(dims):
        cov[i, i] = 1
        cov[i+dims, i+dims] = 1  # First two rows write the eye matrix
        cov[i, i + dims] = rho
        cov[i + dims, i] = rho
    return cov


def generate_gaussian(rho, batch_size, dims, cubic=False):
    """Generate Gaussians"""
    cov = making_cov(rho, dims)
    z = np.random.multivariate_normal(mean=np.repeat(0, dims*2), cov=cov, size=batch_size)
    if cubic:
        z[:, dims:] = z[:, dims:]**3
    return z


class ModelBasicClassification(nn.Module):
    def __init__(self, input_shape, class_size):
        super(ModelBasicClassification, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, class_size)
        )

    def forward(self, x):
        return self.main(x)


def discretize(data, bins):
    """Discretize the data"""
    split = np.array_split(np.sort(data), bins)
    cutoffs = [x[-1] for x in split]
    cutoffs = cutoffs[:-1]
    discrete = np.digitize(data, cutoffs, right=True)
    return discrete, cutoffs


def discretize_batch(data, bins):
    """Discretize the batch."""
    z_disc = np.zeros((data.shape[0], data.shape[1]))
    for d in range(data.shape[1]):
        z_disc[:, d], _ = discretize(data[:, d], bins)
    return z_disc


def njee_staircase(proc_params, opt_params, cubic, latent_dim):
    """Creates the MI staircase for NJEE."""
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    bins = opt_params['batch_size'] - 4
    dims = latent_dim
    loss_fn = nn.CrossEntropyLoss()

    model_lst = []
    opt_lst = []
    for m in range(0, dims):
        if m == 0:
            model_lst.append(None)
            opt_lst.append(None)
        else:
            model_tmp = ModelBasicClassification(m, bins)
            model_tmp.train()
            optim_tmp = optim.Adam(model_tmp.parameters(), lr=5e-4)
            model_lst.append(model_tmp)
            opt_lst.append(optim_tmp)
    model_lst_cond = []
    opt_lst_cond = []
    for m in range(0, dims):
        model_tmp = ModelBasicClassification(dims + m, bins)
        model_tmp.train()
        optim_tmp = optim.Adam(model_tmp.parameters(), lr=5e-4)
        model_lst_cond.append(model_tmp)
        opt_lst_cond.append(optim_tmp)

    r_lst = []
    I = proc_params['levels_MI']#np.arange(2, 12, 2)  # 22
    for i in I:
        r_lst.append((1 - np.exp(-2 * i / dims)) ** 0.5)

    epochs = 4000
    batch_size = opt_params['batch_size']
    H_y_lst = [[] for _ in range(dims)]
    H_yx_lst = [[] for _ in range(dims)]
    I_hat = []
    for r in r_lst:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time)
        print("r: ", r)
        for i in range(epochs):
            z_0 = generate_gaussian(r, batch_size, dims, cubic)
            for j in range(0, dims):
                if j != 0:
                    model_lst[j].train()
                    opt_lst[j].zero_grad()
                    x = torch.Tensor(z_0[:, range(dims, dims + j)])
                    y = z_0[:, dims + j]
                    y = np.reshape(y, [-1, 1])
                    y = torch.Tensor(discretize_batch(y, bins)).long().squeeze()
                    y_pred = model_lst[j](x)
                    loss = loss_fn(y_pred, y)
                    loss.backward()
                    opt_lst[j].step()
                    H_y_lst[j].append(loss.item())
                else:
                    y = z_0[:, j]
                    y = np.reshape(y, [-1, 1])
                    y = discretize_batch(y, bins)
                    _, p_1 = np.unique(y, return_counts=True)
                    p_1 = p_1 / (p_1.sum() + 10 ** -5)
                    H_y_lst[j].append(-np.sum(np.array(p_1) * np.log(p_1)) + (bins - 1) / (2 * batch_size))

                model_lst_cond[j].train()
                opt_lst_cond[j].zero_grad()
                x = torch.Tensor(z_0[:, range(dims + j)])
                y = z_0[:, dims + j]
                y = np.reshape(y, [-1, 1])
                y = torch.Tensor(discretize_batch(y, bins)).long().squeeze()
                y_pred = model_lst_cond[j](x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                opt_lst_cond[j].step()
                H_yx_lst[j].append(loss.item())

            H_y = pd.Series(np.reshape(np.sum(H_y_lst, axis=0), [-1]))
            H_yx = pd.Series(np.reshape(np.sum(H_yx_lst, axis=0), [-1]))
            I_hat.append(H_y.iloc[-1] - H_yx.iloc[-1])
    return I_hat



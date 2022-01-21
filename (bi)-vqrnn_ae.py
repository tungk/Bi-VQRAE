import argparse
import math
import os
import sqlite3
import time
import traceback
import matplotlib as mpl
from sklearn import preprocessing
from torch.autograd import Variable
from utils.outputs import VRAEOutput
mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, cohen_kappa_score, auc, precision_recall_curve, roc_curve, fbeta_score, confusion_matrix
from torch.optim import Adam, lr_scheduler
from units.forget_mult import ForgetMult
from utils.config import VQRAEConfig
from utils.data_provider import read_GD_dataset, generate_synthetic_dataset, read_HSS_dataset, read_S5_dataset, \
    read_NAB_dataset, read_2D_dataset, read_ECG_dataset, read_SMD_dataset, rolling_window_2D, \
    get_loader, cutting_window_2D, unroll_window_3D, read_SMAP_dataset, read_MSL_dataset, read_SWAT_dataset, \
    read_WADI_dataset
from utils.logger import create_logger
from utils.metrics import calculate_average_metric, zscore, create_label_based_on_zscore, \
    MetricsResult, create_label_based_on_quantile
from utils.utils import str2bool
import warnings
warnings.filterwarnings("ignore")
from utils.device import get_free_device


class PlanarTransform(nn.Module):
    def __init__(self, latent_dim=20):
        super(PlanarTransform, self).__init__()
        self.latent_dim = latent_dim
        self.u = nn.Parameter(torch.randn(1, self.latent_dim) * 0.01)
        self.w = nn.Parameter(torch.randn(1, self.latent_dim) * 0.01)
        self.b = nn.Parameter(torch.randn(()) * 0.01)
    def m(self, x):
        return -1 + torch.log(1 + torch.exp(x))
    def h(self, x):
        return torch.tanh(x)
    def h_prime(self, x):
        return 1 - torch.tanh(x) ** 2
    def forward(self, z, logdet=False):
        # z.size() = batch x dim
        u_dot_w = (self.u @ self.w.t()).view(())
        w_hat = self.w / torch.norm(self.w, p=2) # Unit vector in the direction of w
        u_hat = (self.m(u_dot_w) - u_dot_w) * (w_hat) + self.u # 1 x dim
        affine = z @ self.w.t() + self.b
        z_next = z + u_hat * self.h(affine) # batch x dim
        if logdet:
            psi = self.h_prime(affine) * self.w # batch x dim
            LDJ = -torch.log(torch.abs(psi @ u_hat.t() + 1) + 1e-8) # batch x 1
            return z_next, LDJ
        return z_next


class PlanarFlow(nn.Module):
    def __init__(self, latent_dim=20, K=16):
        super(PlanarFlow, self).__init__()
        self.latent_dim = latent_dim
        self.transforms = nn.ModuleList([PlanarTransform(self.latent_dim) for k in range(K)])

    def forward(self, z, logdet=False):
        zK = z
        SLDJ = 0.
        for transform in self.transforms:
            out = transform(zK, logdet=logdet)
            if logdet:
                SLDJ += out[1]
                zK = out[0]
            else:
                zK = out

        if logdet:
            return zK, SLDJ
        return zK


class PlanarFlowSequence(nn.Module):
    def __init__(self, sequence_length, latent_dim, K=10):
        super(PlanarFlowSequence, self).__init__()
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.K = K
        self.transforms_list = nn.ModuleList([PlanarFlow(latent_dim=self.latent_dim, K=self.K) for time_step in range(sequence_length)])

    def forward(self, z_sequence):
        output_K, output_LogDet = [], []
        for time_step in range(self.sequence_length):
            output_k, output_logdet = self.transforms_list[time_step](z_sequence[:, time_step])
            output_K.append(output_k)
            output_LogDet.append(output_logdet)
        return torch.stack(output_K, dim=1), torch.stack(output_LogDet, dim=1)


class QRNNLayer(nn.Module):
    r"""Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.
    Args:
        input_size: The number of expected features in the input x.
        hidden_size: The number of features in the hidden state h. If not specified, the input size is used.
        save_prev_x: Whether to store previous inputs for use in future convolutional windows (i.e. for a continuing sequence such as in language modeling). If true, you must call reset to remove cached previous values of x. Default: False.
        window: Defines the size of the convolutional window (how many previous tokens to look when computing the QRNN values). Supports 1 and 2. Default: 1.
        zoneout: Whether to apply zoneout (i.e. failing to update elements in the hidden state) to the hidden state updates. Default: 0.
        output_gate: If True, performs QRNN-fo (applying an output gate to the output). If False, performs QRNN-f. Default: True.
        use_cuda: If True, uses fast custom CUDA kernel. If False, uses naive for loop. Default: True.
    Inputs: X, hidden
        - X (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - hidden (batch, hidden_size): tensor containing the initial hidden state for the QRNN.
    Outputs: output, h_n
        - output (seq_len, batch, hidden_size): tensor containing the output of the QRNN for each timestep.
        - h_n (batch, hidden_size): tensor containing the hidden state for t=seq_len
    """

    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=1, output_gate=True):
        super(QRNNLayer, self).__init__()

        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate

        # One large matmul with concat is faster than N small matmuls and no concat
        self.linear = nn.Linear(self.window * self.input_size, 3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)

    def reset(self):
        # If you are saving the previous value of x, you should call this when starting with a new state
        self.prevX = None

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()

        source = None
        if self.window == 1:
            source = X
        elif self.window == 2:
            # Construct the x_{t-1} tensor with optional x_{-1}, otherwise a zeroed out value for x_{-1}
            Xm1 = []
            Xm1.append(self.prevX if self.prevX is not None else X[:1, :, :] * 0)
            # Note: in case of len(X) == 1, X[:-1, :, :] results in slicing of empty tensor == bad
            if len(X) > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            # Convert two (seq_len, batch_size, hidden) tensors to (seq_len, batch_size, 2 * hidden)
            source = torch.cat([X, Xm1], 2)

        # Matrix multiplication for the three outputs: Z, F, O
        Y = self.linear(source)
        # Convert the tensor back to (batch, seq_len, len([Z, F, O]) * hidden_size)
        if self.output_gate:
            Y = Y.view(seq_len, batch_size, 3 * self.hidden_size)
            Z, F, O = Y.chunk(3, dim=2)
        else:
            Y = Y.view(seq_len, batch_size, 2 * self.hidden_size)
            Z, F = Y.chunk(2, dim=2)
        ###
        Z = torch.tanh(Z)
        F = torch.sigmoid(F)

        # If zoneout is specified, we perform dropout on the forget gates in F
        # If an element of F is zero, that means the corresponding neuron keeps the old value
        if self.zoneout:
            if self.training:
                mask = Variable(F.data.new(*F.size()).bernoulli_(1 - self.zoneout), requires_grad=False)
                F = F * mask
            else:
                F *= 1 - self.zoneout

        # Ensure the memory is laid out as expected for the CUDA kernel
        # This is a null op if the tensor is already contiguous
        Z = Z.contiguous()
        F = F.contiguous()
        # The O gate doesn't need to be contiguous as it isn't used in the CUDA kernel

        # Forget Mult
        # For testing QRNN without ForgetMult CUDA kernel, C = Z * F may be useful
        C = ForgetMult()(F.to(torch.device("cuda:0")), Z.to(torch.device("cuda:0")), hidden.to(torch.device("cuda:0")))

        # Apply (potentially optional) output gate
        if self.output_gate:
            H = torch.sigmoid(O) * C.to(device)
        else:
            H = C

        # In an optimal world we may want to backprop to x_{t-1} but ...
        if self.window > 1 and self.save_prev_x:
            self.prevX = Variable(X[-1:, :, :].data, requires_grad=False)

        return H, C[-1:, :, :]


class BiDirQRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=1, output_gate=True):
        super(BiDirQRNNLayer, self).__init__()

        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate

        self.forward_qrnn = QRNNLayer(input_size, hidden_size=hidden_size, save_prev_x=save_prev_x, zoneout=zoneout,
                                      window=window, output_gate=output_gate)
        self.backward_qrnn = QRNNLayer(input_size, hidden_size=hidden_size, save_prev_x=save_prev_x, zoneout=zoneout,
                                       window=window, output_gate=output_gate)

    def forward(self, X, hidden=None):
        if not hidden is None:
            fwd, h_fwd = self.forward_qrnn(X, hidden=hidden)
            bwd, h_bwd = self.backward_qrnn(torch.flip(X, [0]), hidden=hidden)
        else:
            fwd, h_fwd = self.forward_qrnn(X)
            bwd, h_bwd = self.backward_qrnn(torch.flip(X, [0]))
        bwd = torch.flip(bwd, [0])
        # return torch.cat([fwd, bwd], dim=-1), torch.cat([h_fwd, h_bwd], dim=-1)
        return fwd, bwd, h_fwd, h_bwd


class VQRAE(nn.Module):
    def __init__(self, file_name, config):
        super(VQRAE, self).__init__()
        # file info
        self.dataset = config.dataset
        self.file_name = file_name

        # dim info
        self.x_dim = config.x_dim
        self.h_dim = config.h_dim
        self.dense_dim = config.h_dim
        self.z_dim = config.z_dim

        # sequence info
        self.preprocessing = config.preprocessing
        self.use_overlapping = config.use_overlapping
        self.use_last_point = config.use_last_point
        self.rolling_size = config.rolling_size

        # optimization info
        self.epochs = config.epochs
        self.milestone_epochs = config.milestone_epochs
        self.lr = config.lr
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.annealing = config.annealing
        self.early_stopping = config.early_stopping
        self.loss_function = config.loss_function
        self.display_epoch = config.display_epoch
        self.lmbda = config.lmbda
        self.use_clip_norm = config.use_clip_norm
        self.gradient_clip_norm = config.gradient_clip_norm

        # dropout
        self.dropout = config.dropout
        self.continue_training = config.continue_training

        self.robustness = config.robustness

        # layers
        self.rnn_layers = config.rnn_layers
        self.use_PNF = config.use_PNF
        self.PNF_layers = config.PNF_layers
        self.use_bidirection = config.use_bidirection
        self.robust_coeff = config.robust_coeff

        # pid
        self.pid = config.pid

        self.save_model = config.save_model
        if self.save_model:
            self.save_model_path = \
                './save_models/{}/VQRAE' \
                '_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.save_model_path = None

        self.load_model = config.load_model
        if self.load_model:
            self.load_model_path = \
                './save_models/{}/VQRAE' \
                '_{}_pid={}.pt'.format(self.dataset, Path(self.file_name).stem, self.pid)
        else:
            self.load_model_path = None

        if self.use_bidirection:
            # encoder  x/u to z, input to latent variable, inference model
            self.phi_enc = nn.Sequential(
                nn.Linear(self.h_dim * 2, self.h_dim * 2),
                nn.ReLU(),
                nn.Linear(self.h_dim * 2, self.h_dim * 2),
                nn.ReLU())
            self.enc_mean = nn.Sequential(
                nn.Linear(self.h_dim * 2, self.z_dim),
                nn.Sigmoid()).to(device)
            self.enc_std = nn.Sequential(
                nn.Linear(self.h_dim * 2, self.z_dim),
                nn.Softplus()).to(device)

            # prior transition of zt-1 to zt
            self.phi_prior = nn.Sequential(
                nn.Linear(self.h_dim * 2, self.h_dim * 2),
                nn.ReLU(),
                nn.Linear(self.h_dim * 2, self.h_dim * 2),
                nn.ReLU()).to(device)
            self.prior_mean = nn.Sequential(
                nn.Linear(self.h_dim * 2, self.z_dim),
                nn.Sigmoid()).to(device)
            self.prior_std = nn.Sequential(
                nn.Linear(self.h_dim * 2, self.z_dim),
                nn.Softplus()).to(device)

            # decoder
            self.phi_dec = nn.Sequential(
                nn.Linear(self.h_dim * 2 + self.z_dim, self.h_dim * 2),
                nn.ReLU(),
                nn.Linear(self.h_dim * 2, self.h_dim * 2),
                nn.ReLU()).to(device)
            self.dec_std = nn.Sequential(
                nn.Linear(self.h_dim * 2, self.x_dim),
                nn.Softplus()).to(device)
            self.dec_mean = nn.Sequential(
                nn.Linear(self.h_dim * 2, self.x_dim),
                nn.Sigmoid()).to(device)

            self.forward_hidden_state_qrnn = torch.nn.ModuleList(
                [QRNNLayer(input_size=self.x_dim if l == 0 else self.h_dim, hidden_size=self.h_dim)
                 for l in range(self.rnn_layers)]).to(device)
            self.forward_qrnn = torch.nn.ModuleList(
                [QRNNLayer(input_size=self.x_dim + self.h_dim if l == 0 else self.h_dim, hidden_size=self.h_dim)
                 for l in range(self.rnn_layers)]).to(device)
            self.backward_hidden_state_qrnn = torch.nn.ModuleList(
                [QRNNLayer(input_size=self.x_dim if l == 0 else self.h_dim, hidden_size=self.h_dim)
                 for l in range(self.rnn_layers)]).to(device)
            self.backward_qrnn = torch.nn.ModuleList(
                [QRNNLayer(input_size=self.x_dim + self.h_dim if l == 0 else self.h_dim, hidden_size=self.h_dim)
                 for l in range(self.rnn_layers)]).to(device)
        else:
            # encoder  x/u to z, input to latent variable, inference model
            self.phi_enc = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU())
            self.enc_mean = nn.Sequential(
                nn.Linear(self.h_dim, self.z_dim),
                nn.Sigmoid()).to(device)
            self.enc_std = nn.Sequential(
                nn.Linear(self.h_dim, self.z_dim),
                nn.Softplus()).to(device)

            # prior transition of zt-1 to zt
            self.phi_prior = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU()).to(device)
            self.prior_mean = nn.Sequential(
                nn.Linear(self.h_dim, self.z_dim),
                nn.Sigmoid()).to(device)
            self.prior_std = nn.Sequential(
                nn.Linear(self.h_dim, self.z_dim),
                nn.Softplus()).to(device)

            # decoder
            self.phi_dec = nn.Sequential(
                nn.Linear(self.h_dim + self.z_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU()).to(device)
            self.dec_std = nn.Sequential(
                nn.Linear(self.h_dim, self.x_dim),
                nn.Softplus()).to(device)
            self.dec_mean = nn.Sequential(
                nn.Linear(self.h_dim, self.x_dim),
                nn.Sigmoid()).to(device)

            self.hidden_state_qrnn = torch.nn.ModuleList(
                [QRNNLayer(input_size=self.x_dim if l == 0 else self.h_dim, hidden_size=self.h_dim)
                 for l in range(self.rnn_layers)]).to(device)
            self.qrnn = torch.nn.ModuleList(
                [QRNNLayer(input_size=self.x_dim + self.h_dim if l == 0 else self.h_dim, hidden_size=self.h_dim)
                 for l in range(self.rnn_layers)]).to(device)

        if self.use_PNF:
            # self.PNF = PlanarFlowSequence(sequence_length=self.rolling_size, latent_dim=self.z_dim, K=self.PNF_layers).to(device)
            self.PNF = PlanarFlow(latent_dim=self.z_dim, K=self.PNF_layers).to(device)

    def reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(device)
        return eps.mul(std).add_(mean)

    def beta_bernoulli(self, mean, x, beta):
        term1 = (1 / beta)
        mean = mean * 0.999999 + 1e-6  ###for avoiding numerical error
        term2 = (x * torch.pow(mean, beta)) + (1 - x) * torch.pow((1 - mean), beta)
        term2 = torch.prod(term2, dim=1) - 1
        term3 = torch.pow(mean, (beta + 1)) + torch.pow((1 - mean), (beta + 1))
        term3 = torch.prod(term3, dim=1) / (beta + 1)
        loss = torch.sum(-term1 * term2 + term3)

        if torch.isnan(loss):
            print('nan loss')

        return loss

    def beta_gaussian_1(self, mean, x, beta, sigma=0.5):
        D = mean.shape[1]
        term1 = -((1 + beta) / beta)
        K1 = 1 / pow((2 * math.pi * (sigma ** 2)), (beta * D / 2))
        term2 = torch.nn.functional.mse_loss(input=mean, target=x, reduction='mean')
        term3 = torch.exp(-(beta / (2 * (sigma ** 2))) * term2)
        loss = torch.sum(term1 * (K1 * term3 - 1))
        return loss

    def beta_gaussian_2(self, mean, std, x, beta, sigma=0.5):
        D = mean.shape[1]
        term1 = -((1 + beta) / beta)
        K1 = 1 / pow((2 * math.pi * (sigma ** 2)), (beta * D / 2))
        term2 = 0.5 * (torch.sum(std) + torch.sum(((x - mean) / std.mul(0.5).exp_()) ** 2))
        # term2 = torch.nn.functional.mse_loss(input=mean, target=x, reduction='sum')
        term3 = torch.exp(-(beta / (2 * (sigma ** 2))) * term2)
        loss = torch.sum(term1 * (K1 * term3 - 1))
        return loss

    def nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def nll_gaussian_1(self, mean, std, x):
        return 0.5 * (torch.sum(std) + torch.sum(((x - mean) / std.mul(0.5).exp_()) ** 2))  # Owned definition

    def nll_gaussian_2(self, mean, std, x):
        return torch.sum(torch.log(std) + (x - mean).pow(2) / (2 * std.pow(2)))

    def mse(self, mean, x):
        return torch.nn.functional.mse_loss(input=mean, target=x, reduction='mean')

    def kld_gaussian(self, mean_1, std_1, mean_2, std_2):
        if mean_2 is not None and std_2 is not None:
            kl_loss = 0.5 * torch.sum(
                2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(
                    2) - 1)
        else:
            kl_loss = -0.5 * torch.sum(1 + std_1 - mean_1.pow(2) - std_1.exp())
        return kl_loss

    def kld_gaussian_w_logdet(self, mean_1, std_1, mean_2, std_2, z_0, z_k, SLDJ):
        if mean_2 is not None and std_2 is not None:
            # q0 = torch.distributions.normal.Normal(mean_1, (0.5 * std_1).exp())
            q0 = torch.distributions.normal.Normal(mean_1, std_1)
            prior = torch.distributions.normal.Normal(mean_2, std_2)
            log_prior_zK = prior.log_prob(z_k).sum(-1)
            log_q0_z0 = q0.log_prob(z_0).sum(-1)
            log_q0_zK = log_q0_z0 + SLDJ.sum(-1)
            kld = (log_q0_zK - log_prior_zK).sum()
            return kld
        else:
            # q0 = torch.distributions.normal.Normal(mean_1, (0.5 * std_1).exp())
            q0 = torch.distributions.normal.Normal(mean_1, std_1)
            prior = torch.distributions.normal.Normal(0., 1.)
            log_prior_zK = prior.log_prob(z_k).sum(-1)
            log_q0_z0 = q0.log_prob(z_0).sum(-1)
            log_q0_zK = log_q0_z0 + SLDJ.sum(-1)
            kld = (log_q0_zK - log_prior_zK).sum()
            return kld

    def kld_gaussian_1(self, mean, std):
        """Using std to compute KLD"""
        return -0.5 * torch.sum(1 + torch.log(std) - mean.pow(2) - std)

    def kld_gaussian_2(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) + (std_1.pow(2) + (mean_1 - mean_2).pow(2)) / std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def kl_anneal_function(self, anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return min(1, step / x0)

    def forward(self, x, y, hidden=None):
        if self.use_bidirection:
            fh_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(device)
            bh_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(device)
            fh, fh_t = self.forward_hidden_state_qrnn[0](x, fh_0)
            bh, bh_t = self.backward_hidden_state_qrnn[0](torch.flip(x, [0]), bh_0)

            # reversing hidden state list
            reversed_fh = torch.flip(fh, dims=[0])
            reversed_bh = torch.flip(bh, dims=[0])

            # reversing y_t list
            reversed_y = torch.flip(y, dims=[0])
            original_y = y

            # concat reverse h with reverse x_t
            concat_fh_ry = torch.cat([reversed_y, reversed_fh], dim=2)
            concat_bh_oy = torch.cat([original_y, reversed_bh], dim=2)

            # compute reverse a_t
            fa_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(device)
            ba_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(device)
            fa, fa_t = self.forward_qrnn[0](concat_fh_ry, fa_0)
            ba, ba_t = self.backward_qrnn[0](concat_bh_oy, ba_0)

            reversed_fa = torch.flip(fa, dims=[0])
            reversed_ba = torch.flip(ba, dims=[0])

            enc = self.phi_enc(torch.cat([reversed_fa, reversed_ba], dim=2).permute(1, 0, 2)).unsqueeze(-2)
            enc_mean = self.enc_mean(enc).squeeze(2)
            enc_std = self.enc_std(enc).squeeze(2)
            z_0 = self.reparameterized_sample(enc_mean, enc_std)

            if self.use_PNF:
                z_k, logdet = self.PNF(z_0, True)

            prior = self.phi_prior(torch.cat([fh, bh], dim=2).permute(1, 0, 2)).unsqueeze(-2)
            prior_mean = self.prior_mean(prior).squeeze(2)
            prior_std = self.prior_std(prior).squeeze(2)

            if self.use_PNF:
                dec = self.phi_dec(torch.cat([z_k, fh.permute(1, 0, 2), bh.permute(1, 0, 2)], dim=2)).unsqueeze(-2)
            else:
                dec = self.phi_dec(torch.cat([z_0, fh.permute(1, 0, 2), bh.permute(1, 0, 2)], dim=2)).unsqueeze(-2)

            dec_mean = self.dec_mean(dec).squeeze(2)
            dec_std = self.dec_std(dec).squeeze(2)

            if self.use_PNF:
                kld_loss = self.kld_gaussian_w_logdet(mean_1=enc_mean, std_1=enc_std, mean_2=prior_mean, std_2=prior_std, z_0=z_0, z_k=z_k, SLDJ=logdet)
            else:
                kld_loss = self.kld_gaussian(mean_1=enc_mean, std_1=enc_std, mean_2=prior_mean, std_2=prior_std)
                # kld_loss = self.kld_gaussian(mean_1=enc_mean, std_1=enc_std, mean_2=None, std_2=None)
            if self.loss_function == 'nll':
                nll_loss = self.nll_gaussian_1(mean=dec_mean, std=dec_std, x=y.permute(1, 0, 2))
            elif self.loss_function == 'mse':
                nll_loss = self.mse(mean=dec_mean, x=y.permute(1, 0, 2))
            elif self.loss_function == 'beta_1':
                nll_loss = self.beta_gaussian_1(mean=dec_mean, x=y.permute(1, 0, 2), beta=self.robust_coeff)
            elif self.loss_function == 'beta_2':
                nll_loss = self.beta_gaussian_2(mean=dec_mean, std=dec_std, x=y.permute(1, 0, 2), beta=self.robust_coeff)


            # return nll_loss, kld_loss, torch.zeros(enc.squeeze(2).shape).to(device), enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(device), dec_mean, dec_std
            if self.use_PNF:
                return nll_loss, kld_loss, z_k, enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(device), dec_mean, dec_std
            else:
                return nll_loss, kld_loss, z_0, enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(device), dec_mean, dec_std

        else:
            # computing hidden state in list and x_t & y_t in list outside the loop
            h_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(device)
            h, h_t = self.hidden_state_qrnn[0](x, h_0)

            # reversing hidden state list
            reversed_h = torch.flip(h, dims=[0])

            # reversing y_t list
            reversed_y = torch.flip(y, dims=[0])

            # concat reverse h with reverse x_t
            concat_h_y = torch.cat([reversed_y, reversed_h], dim=2)

            # compute reverse a_t
            a_0 = Variable(torch.zeros(x.shape[1], self.h_dim), requires_grad=True).to(device)
            a, a_t = self.qrnn[0](concat_h_y, a_0)
            reversed_a = torch.flip(a, dims=[0])

            enc = self.phi_enc(reversed_a.permute(1, 0, 2)).unsqueeze(-2)
            enc_mean = self.enc_mean(enc).squeeze(2)
            enc_std = self.enc_std(enc).squeeze(2)
            z_0 = self.reparameterized_sample(enc_mean, enc_std)

            if self.use_PNF:
                z_k, logdet = self.PNF(z_0, True)

            prior = self.phi_prior(h.permute(1, 0, 2)).unsqueeze(-2)
            prior_mean = self.prior_mean(prior).squeeze(2)
            prior_std = self.prior_std(prior).squeeze(2)

            if self.use_PNF:
                dec = self.phi_dec(torch.cat([z_k, h.permute(1, 0, 2)], dim=2)).unsqueeze(-2)
            else:
                dec = self.phi_dec(torch.cat([z_0, h.permute(1, 0, 2)], dim=2)).unsqueeze(-2)

            dec_mean = self.dec_mean(dec).squeeze(2)
            dec_std = self.dec_std(dec).squeeze(2)

            if self.use_PNF:
                kld_loss = self.kld_gaussian_w_logdet(mean_1=enc_mean, std_1=enc_std, mean_2=prior_mean, std_2=prior_std, z_0=z_0, z_k=z_k, SLDJ=logdet)
            else:
                kld_loss = self.kld_gaussian(mean_1=enc_mean, std_1=enc_std, mean_2=prior_mean, std_2=prior_std)
                # kld_loss = self.kld_gaussian(mean_1=enc_mean, std_1=enc_std, mean_2=None, std_2=None)
            if self.loss_function == 'nll':
                nll_loss = self.nll_gaussian_1(mean=dec_mean, std=dec_std, x=y.permute(1, 0, 2))
            elif self.loss_function == 'mse':
                nll_loss = self.mse(mean=dec_mean, x=y.permute(1, 0, 2))
            elif self.loss_function == 'beta_1':
                nll_loss = self.beta_gaussian_1(mean=dec_mean, x=y.permute(1, 0, 2), beta=self.robust_coeff)
            elif self.loss_function == 'beta_2':
                nll_loss = self.beta_gaussian_2(mean=dec_mean, std=dec_std, x=y.permute(1, 0, 2), beta=self.robust_coeff)

            # return nll_loss, kld_loss, torch.zeros(enc.squeeze(2).shape).to(device), enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(device), dec_mean, dec_std
            if self.use_PNF:
                return nll_loss, kld_loss, z_k, enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(device), dec_mean, dec_std
            else:
                return nll_loss, kld_loss, z_0, enc_mean, enc_std, torch.zeros(dec.squeeze(2).shape).to(device), dec_mean, dec_std

    def fit(self, train_input, train_label, test_input, test_label, abnormal_data, abnormal_label, original_x_dim):
        TN = []
        TP = []
        FN = []
        FP = []
        PRECISION = []
        RECALL = []
        FBETA = []
        PR_AUC = []
        ROC_AUC = []
        CKS = []
        opt = Adam(list(self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        sched = lr_scheduler.StepLR(optimizer=opt, step_size=self.milestone_epochs, gamma=self.gamma)
        # get batch data
        train_data = get_loader(input=train_input, label=train_label, batch_size=self.batch_size, from_numpy=True,
                                drop_last=False, shuffle=False)
        test_data = get_loader(input=test_input, label=test_label, batch_size=self.batch_size, from_numpy=True,
                               drop_last=False, shuffle=False)
        if self.load_model == True and self.continue_training == False:
            self.load_state_dict(torch.load(self.load_model_path))
        elif self.load_model == True and self.continue_training == True:
            start_training_time = time.time()
            self.load_state_dict(torch.load(self.load_model_path))
            # train model
            self.train()
            epoch_losses = []
            epoch_nll_losses = []
            epoch_kld_losses = []
            for epoch in range(self.epochs):
                batch_train_losses = []
                batch_nll_losses = []
                batch_kld_losses = []
                # opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    opt.zero_grad()
                    nll_loss, kld_loss, batch_z, batch_z_mean, batch_z_std, batch_x_reconstruct, batch_x_mean, batch_x_std = self.forward(x=batch_x.permute(1, 0, 2).to(device), y=batch_x.permute(1, 0, 2).to(device))
                    # batch_loss = nll_loss + self.lmbda * epoch * kld_loss
                    if self.annealing:
                        kld_loss = min(1, epoch / 150) if epoch != 0 else 0 * kld_loss
                    else:
                        kld_loss = self.lmbda * kld_loss
                    batch_loss = nll_loss + kld_loss
                    batch_loss.backward()
                    if self.use_clip_norm:
                        torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.gradient_clip_norm)
                    opt.step()
                    sched.step()
                    batch_nll_losses.append(nll_loss.item())
                    batch_kld_losses.append(kld_loss.item())
                    batch_train_losses.append(batch_loss.item())
                epoch_losses.append(mean(batch_train_losses))
                epoch_nll_losses.append(mean(batch_nll_losses))
                epoch_kld_losses.append(mean(batch_kld_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {} , nll loss = {}, kld loss = {}'.format(epoch, epoch_losses[-1], epoch_nll_losses[-1], epoch_kld_losses[-1]))

                    self.eval()
                    with torch.no_grad():
                        cat_zs = []
                        cat_z_means = []
                        cat_z_stds = []
                        cat_xs = []
                        cat_x_means = []
                        cat_x_stds = []
                        kld_loss = 0
                        nll_loss = 0

                        for i, (batch_x, batch_y) in enumerate(test_data):
                            nll_loss, kld_loss, batch_z, batch_z_mean, batch_z_std, batch_x_reconstruct, \
                            batch_x_mean, batch_x_std = self.forward(x=batch_x.permute(1, 0, 2).to(device), y=batch_x.permute(1, 0, 2).to(device))
                            cat_zs.append(batch_z)
                            cat_z_means.append(batch_z_mean)
                            cat_z_stds.append(batch_z_std)
                            cat_xs.append(batch_x_reconstruct)
                            cat_x_means.append(batch_x_mean)
                            cat_x_stds.append(batch_x_std)
                            kld_loss += kld_loss
                            nll_loss += nll_loss

                        cat_zs = torch.cat(cat_zs, dim=0)
                        cat_z_means = torch.cat(cat_z_means, dim=0)
                        cat_z_stds = torch.cat(cat_z_stds, dim=0)
                        cat_xs = torch.cat(cat_xs, dim=0)
                        cat_x_means = torch.cat(cat_x_means, dim=0)
                        cat_x_stds = torch.cat(cat_x_stds, dim=0)

                        vrae_output = VRAEOutput(best_TN=None, best_FP=None, best_FN=None, best_TP=None,
                                                 best_precision=None, best_recall=None, best_fbeta=None,
                                                 best_pr_auc=None, best_roc_auc=None, best_cks=None,
                                                 training_time=0, testing_time=0, zs=cat_zs,
                                                 z_infer_means=cat_z_means, z_infer_stds=cat_z_stds, decs=cat_xs,
                                                 dec_means=cat_x_means, dec_stds=cat_x_stds, kld_loss=kld_loss,
                                                 nll_loss=nll_loss)


                        min_max_scaler = preprocessing.MinMaxScaler()
                        if self.preprocessing:
                            if self.use_overlapping:
                                if self.use_last_point:
                                    dec_mean_unroll = vrae_output.dec_means.detach().cpu().numpy()[:, -1]
                                    dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                                    x_original_unroll = abnormal_data[self.rolling_size - 1:]
                                    abnormal_segment = abnormal_label[self.rolling_size-1:]
                                else:
                                    dec_mean_unroll = unroll_window_3D(np.reshape(vrae_output.dec_means.detach().cpu().numpy(), (-1, self.rolling_size, original_x_dim)))[::-1]
                                    dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                                    x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                                    abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]

                            else:
                                dec_mean_unroll = np.reshape(vrae_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
                                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                                abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]
                        else:
                            dec_mean_unroll = vrae_output.dec_means.detach().cpu().numpy()
                            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                            x_original_unroll = abnormal_data
                            abnormal_segment = abnormal_label

                        error = np.sum(x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim]) ** 2, axis=1)
                        final_zscore = zscore(error)
                        np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)

                        pos_label = -1
                        cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
                        precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                        recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                        fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=0.5)
                        fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_segment), y_score=np.nan_to_num(error), pos_label=pos_label)
                        roc_auc = auc(fpr, tpr)
                        pre, re, _ = precision_recall_curve(y_true=np.nan_to_num(abnormal_segment), probas_pred=np.nan_to_num(error), pos_label=pos_label)
                        pr_auc = auc(re, pre)
                        cks = cohen_kappa_score(y1=abnormal_segment, y2=np_decision)
                        TN.append(cm[0][0])
                        FP.append(cm[0][1])
                        FN.append(cm[1][0])
                        TP.append(cm[1][1])
                        PRECISION.append(precision)
                        RECALL.append(recall)
                        FBETA.append(fbeta)
                        PR_AUC.append(pr_auc)
                        ROC_AUC.append(roc_auc)
                        CKS.append(cks)

                if self.early_stopping:
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break
            end_training_time = time.time()
            elapse_training_time = end_training_time - start_training_time
        else:
            start_training_time = time.time()
            # train model
            self.train()
            epoch_losses = []
            epoch_nll_losses = []
            epoch_kld_losses = []
            for epoch in range(self.epochs):
                batch_train_losses = []
                batch_nll_losses = []
                batch_kld_losses = []
                # opt.zero_grad()
                for i, (batch_x, batch_y) in enumerate(train_data):
                    # All outputs have shape [sequence_length|rolling_size, batch_size, h_d1im|z_dim]
                    opt.zero_grad()
                    nll_loss, kld_loss, batch_z, batch_z_mean, batch_z_std, batch_x_reconstruct, batch_x_mean, batch_x_std = self.forward(x=batch_x.permute(1, 0, 2).to(device), y=batch_x.permute(1, 0, 2).to(device))
                    if self.annealing:
                        kld_loss = min(1, epoch/150) * kld_loss if epoch != 0 else 0 * kld_loss
                    else:
                        kld_loss = self.lmbda * kld_loss
                    batch_loss = nll_loss + kld_loss
                    batch_loss.backward()
                    if self.use_clip_norm:
                        torch.nn.utils.clip_grad_norm_(list(self.parameters()), self.gradient_clip_norm)
                    opt.step()
                    batch_nll_losses.append(nll_loss.item())
                    batch_kld_losses.append(kld_loss.item())
                    batch_train_losses.append(batch_loss.item())
                sched.step()
                epoch_losses.append(mean(batch_train_losses))
                epoch_nll_losses.append(mean(batch_nll_losses))
                epoch_kld_losses.append(mean(batch_kld_losses))
                if epoch % self.display_epoch == 0:
                    train_logger.info('epoch = {} , train loss = {} , nll loss = {}, kld loss = {}'.format(epoch, epoch_losses[-1], epoch_nll_losses[-1], epoch_kld_losses[-1]))

                    self.eval()
                    with torch.no_grad():
                        cat_zs = []
                        cat_z_means = []
                        cat_z_stds = []
                        cat_xs = []
                        cat_x_means = []
                        cat_x_stds = []
                        kld_loss = 0
                        nll_loss = 0

                        for i, (batch_x, batch_y) in enumerate(test_data):
                            nll_loss, kld_loss, batch_z, batch_z_mean, batch_z_std, batch_x_reconstruct, \
                            batch_x_mean, batch_x_std = self.forward(x=batch_x.permute(1, 0, 2).to(device),
                                                                     y=batch_x.permute(1, 0, 2).to(device))
                            cat_zs.append(batch_z)
                            cat_z_means.append(batch_z_mean)
                            cat_z_stds.append(batch_z_std)
                            cat_xs.append(batch_x_reconstruct)
                            cat_x_means.append(batch_x_mean)
                            cat_x_stds.append(batch_x_std)
                            kld_loss += kld_loss
                            nll_loss += nll_loss

                        cat_zs = torch.cat(cat_zs, dim=0)
                        cat_z_means = torch.cat(cat_z_means, dim=0)
                        cat_z_stds = torch.cat(cat_z_stds, dim=0)
                        cat_xs = torch.cat(cat_xs, dim=0)
                        cat_x_means = torch.cat(cat_x_means, dim=0)
                        cat_x_stds = torch.cat(cat_x_stds, dim=0)

                        vrae_output = VRAEOutput(best_TN=None, best_FP=None, best_FN=None, best_TP=None,
                                                 best_precision=None, best_recall=None, best_fbeta=None,
                                                 best_pr_auc=None, best_roc_auc=None, best_cks=None,
                                                 training_time=0, testing_time=0, zs=cat_zs,
                                                 z_infer_means=cat_z_means, z_infer_stds=cat_z_stds, decs=cat_xs,
                                                 dec_means=cat_x_means, dec_stds=cat_x_stds, kld_loss=kld_loss,
                                                 nll_loss=nll_loss)

                        min_max_scaler = preprocessing.MinMaxScaler()
                        if self.preprocessing:
                            if self.use_overlapping:
                                if self.use_last_point:
                                    dec_mean_unroll = vrae_output.dec_means.detach().cpu().numpy()[:, -1]
                                    dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                                    x_original_unroll = abnormal_data[self.rolling_size - 1:]
                                    abnormal_segment = abnormal_label[self.rolling_size-1:]
                                else:
                                    dec_mean_unroll = unroll_window_3D(np.reshape(vrae_output.dec_means.detach().cpu().numpy(), (-1, self.rolling_size, original_x_dim)))[::-1]
                                    dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                                    x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                                    abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]

                            else:
                                dec_mean_unroll = np.reshape(vrae_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
                                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                                abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]
                        else:
                            dec_mean_unroll = vrae_output.dec_means.detach().cpu().numpy()
                            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                            x_original_unroll = abnormal_data
                            abnormal_segment = abnormal_label

                        error = np.sum(x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim]) ** 2, axis=1)
                        final_zscore = zscore(error)
                        np_decision = create_label_based_on_zscore(final_zscore, 2.5, True)

                        pos_label = -1
                        cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
                        precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                        recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
                        fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=0.5)
                        fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_segment), y_score=np.nan_to_num(error), pos_label=pos_label)
                        roc_auc = auc(fpr, tpr)
                        pre, re, _ = precision_recall_curve(y_true=np.nan_to_num(abnormal_segment), probas_pred=np.nan_to_num(error), pos_label=pos_label)
                        pr_auc = auc(re, pre)
                        cks = cohen_kappa_score(y1=abnormal_segment, y2=np_decision)
                        TN.append(cm[0][0])
                        FP.append(cm[0][1])
                        FN.append(cm[1][0])
                        TP.append(cm[1][1])
                        PRECISION.append(precision)
                        RECALL.append(recall)
                        FBETA.append(fbeta)
                        PR_AUC.append(pr_auc)
                        ROC_AUC.append(roc_auc)
                        CKS.append(cks)

                if self.early_stopping:
                    if epoch > 1:
                        if -1e-7 < epoch_losses[-1] - epoch_losses[-2] < 1e-7:
                            train_logger.info('early break')
                            break
            end_training_time = time.time()
            elapse_training_time = end_training_time - start_training_time

        # ICDE revision (KL visualization)
        np.savetxt('./save_outputs/{}/VQRAE_NLL' \
                '_{}_pid={}.txt'.format(self.dataset, Path(self.file_name).stem, self.pid), np.asarray(epoch_nll_losses))
        np.savetxt('./save_outputs/{}/VQRAE_KLD' \
                   '_{}_pid={}.txt'.format(self.dataset, Path(self.file_name).stem, self.pid), np.asarray(epoch_kld_losses))

        if self.save_model:
            torch.save(self.state_dict(), self.save_model_path)
        # test model
        self.eval()
        with torch.no_grad():
            cat_zs = []
            cat_z_means = []
            cat_z_stds = []
            cat_xs = []
            cat_x_means = []
            cat_x_stds = []
            kld_loss = 0
            nll_loss = 0

            for i, (batch_x, batch_y) in enumerate(test_data):
                nll_loss, kld_loss, batch_z, batch_z_mean, batch_z_std, batch_x_reconstruct, batch_x_mean, batch_x_std = self.forward(x=batch_x.permute(1, 0, 2).to(device), y=batch_x.permute(1, 0, 2).to(device))
                cat_zs.append(batch_z)
                cat_z_means.append(batch_z_mean)
                cat_z_stds.append(batch_z_std)
                cat_xs.append(batch_x_reconstruct)
                cat_x_means.append(batch_x_mean)
                cat_x_stds.append(batch_x_std)
                kld_loss += kld_loss
                nll_loss += nll_loss

            cat_zs = torch.cat(cat_zs, dim=0)
            cat_z_means = torch.cat(cat_z_means, dim=0)
            cat_z_stds = torch.cat(cat_z_stds, dim=0)
            cat_xs = torch.cat(cat_xs, dim=0)
            cat_x_means = torch.cat(cat_x_means, dim=0)
            cat_x_stds = torch.cat(cat_x_stds, dim=0)

            if len(TN) == 0:
                TN.append(0)
            if len(FP) == 0:
                FP.append(0)
            if len(FN) == 0:
                FN.append(0)
            if len(TP) == 0:
                TP.append(0)
            if len(PRECISION) == 0:
                PRECISION.append(0)
            if len(RECALL) == 0:
                RECALL.append(0)
            if len(FBETA) == 0:
                FBETA.append(0)
            if len(PR_AUC) == 0:
                PR_AUC.append(0)
            if len(ROC_AUC) == 0:
                ROC_AUC.append(0)
            if len(CKS) == 0:
                CKS.append(0)

            vrae_output = VRAEOutput(best_TN=max(TN), best_FP=max(FP), best_FN=max(FN), best_TP=max(TP),
                                     best_precision=max(PRECISION), best_recall=max(RECALL), best_fbeta=max(FBETA),
                                     best_pr_auc=max(PR_AUC), best_roc_auc=max(ROC_AUC), best_cks=max(CKS), 
                                     training_time=0, testing_time=0, zs=cat_zs,
                                     z_infer_means=cat_z_means, z_infer_stds=cat_z_stds, decs=cat_xs,
                                     dec_means=cat_x_means, dec_stds=cat_x_stds, kld_loss=kld_loss, nll_loss=nll_loss)
            return vrae_output

def RunModel(file_name, config):
    train_data = None
    if config.dataset == 0:
        abnormal_data, abnormal_label = generate_synthetic_dataset(case=0, length=args.length, max=args.max,
                                                                   theta=args.theta, anomalies=args.anomalies,
                                                                   noise=False, verbose=False)
    if config.dataset == 1:
        abnormal_data, abnormal_label = read_GD_dataset(file_name)
    if config.dataset == 2:
        abnormal_data, abnormal_label = read_HSS_dataset(file_name)
    if config.dataset == 3 or config.dataset == 31 or config.dataset == 32 or config.dataset == 33 or config.dataset == 34 or config.dataset == 35:
        abnormal_data, abnormal_label = read_S5_dataset(file_name)
    if config.dataset == 4 or config.dataset == 41 or config.dataset == 42 or config.dataset == 43 or config.dataset == 44 or config.dataset == 45 or config.dataset == 46:
        abnormal_data, abnormal_label = read_NAB_dataset(file_name)
    if config.dataset == 5 or config.dataset == 51 or config.dataset == 52 or config.dataset == 53 or config.dataset == 54 or config.dataset == 55 or config.dataset == 56 or config.dataset == 57:
        train_data, abnormal_data, abnormal_label = read_2D_dataset(file_name)
    if config.dataset == 6 or config.dataset == 61 or config.dataset == 62 or config.dataset == 63 or config.dataset == 64 or config.dataset == 65 or config.dataset == 66 or config.dataset == 67:
        abnormal_data, abnormal_label = read_ECG_dataset(file_name)
    if config.dataset == 7 or config.dataset == 71 or config.dataset == 72 or config.dataset == 73:
        train_data, abnormal_data, abnormal_label = read_SMD_dataset(file_name)
    if config.dataset == 8 or config.dataset == 81 or config.dataset == 82 or config.dataset == 83 or config.dataset == 84 or config.dataset == 85 or config.dataset == 86 or config.dataset == 87 or config.dataset == 88 or config.dataset == 89 or config.dataset == 90:
        train_data, abnormal_data, abnormal_label = read_SMAP_dataset(file_name)
    if config.dataset == 9 or config.dataset == 91 or config.dataset == 92 or config.dataset == 93 or config.dataset == 94 or config.dataset == 95 or config.dataset == 96 or config.dataset == 97:
        train_data, abnormal_data, abnormal_label = read_MSL_dataset(file_name)
    if config.dataset == 101:
        train_data, abnormal_data, abnormal_label = read_SWAT_dataset(file_name)
    if config.dataset == 11 or config.dataset == 111 or config.dataset == 112:
        train_data, abnormal_data, abnormal_label = read_WADI_dataset(file_name)

    original_x_dim = abnormal_data.shape[1]

    if config.preprocessing:
        if config.use_overlapping:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(train_data, config.rolling_size), rolling_window_2D(abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
            else:
                rolling_abnormal_data, rolling_abnormal_label = rolling_window_2D(abnormal_data, config.rolling_size), rolling_window_2D(abnormal_label, config.rolling_size)
        else:
            if train_data is not None:
                rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(train_data, config.rolling_size), cutting_window_2D(abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
            else:
                rolling_abnormal_data, rolling_abnormal_label = cutting_window_2D(abnormal_data, config.rolling_size), cutting_window_2D(abnormal_label, config.rolling_size)
    else:
        if train_data is not None:
            rolling_train_data, rolling_abnormal_data, rolling_abnormal_label = train_data, abnormal_data, abnormal_label
        else:
            rolling_abnormal_data, rolling_abnormal_label = abnormal_data, abnormal_label

    config.x_dim = rolling_abnormal_data.shape[2]
    model = VQRAE(file_name=file_name, config=config)
    model = model.to(device)
    if train_data is not None and config.robustness == False:
        vqrae_output = model.fit(train_input=rolling_train_data, train_label=rolling_train_data,
                                 test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                                 abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                                 original_x_dim=original_x_dim)
    elif train_data is None or config.robustness == True:
        vqrae_output = model.fit(train_input=rolling_abnormal_data, train_label=rolling_abnormal_data,
                                 test_input=rolling_abnormal_data, test_label=rolling_abnormal_label,
                                 abnormal_data=abnormal_data, abnormal_label=abnormal_label,
                                 original_x_dim=original_x_dim)
    # %%
    execution_time = time.time() - start_time_file
    min_max_scaler = preprocessing.StandardScaler()
    if config.preprocessing:
        if config.use_overlapping:
            if config.use_last_point:
                dec_mean_unroll = vqrae_output.dec_means.detach().cpu().numpy()[:, -1]
                dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                latent_mean_unroll = vqrae_output.zs.detach().cpu().numpy()
                # x_original_unroll = abnormal_data[config.rolling_size - 1:]
                x_original_unroll = rolling_abnormal_data[:, -1]
                x_original_unroll = min_max_scaler.fit_transform(x_original_unroll)
                # abnormal_segment = abnormal_label[config.rolling_size - 1:]
                abnormal_segment = rolling_abnormal_label[:, -1]
            else:
                dec_mean_unroll = unroll_window_3D(np.reshape(vqrae_output.dec_means.detach().cpu().numpy(), (-1, config.rolling_size, original_x_dim)))[::-1]  # check the reverse here
                # dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
                latent_mean_unroll = vqrae_output.zs.detach().cpu().numpy()
                x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
                # x_original_unroll = min_max_scaler.fit_transform(x_original_unroll)
                abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]

        else:
            dec_mean_unroll = np.reshape(vqrae_output.dec_means.detach().cpu().numpy(), (-1, original_x_dim))
            dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
            latent_mean_unroll = vqrae_output.zs.detach().cpu().numpy()
            x_original_unroll = abnormal_data[: dec_mean_unroll.shape[0]]
            x_original_unroll = min_max_scaler.fit_transform(x_original_unroll)
            abnormal_segment = abnormal_label[: dec_mean_unroll.shape[0]]
    else:
        dec_mean_unroll = vqrae_output.dec_means.detach().cpu().numpy()
        dec_mean_unroll = min_max_scaler.fit_transform(dec_mean_unroll)
        latent_mean_unroll = vqrae_output.zs.detach().cpu().numpy()
        x_original_unroll = abnormal_data
        x_original_unroll = min_max_scaler.fit_transform(x_original_unroll)
        abnormal_segment = abnormal_label

    # %%
    if config.save_output:
        np.save('./save_outputs/NPY/{}/Dec_VQRAE_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), dec_mean_unroll)
        np.save('./save_outputs/NPY/{}/Latent_VQRAE_{}_pid={}.npy'.format(config.dataset, Path(file_name).stem, config.pid), latent_mean_unroll)


    error = np.sum((x_original_unroll - np.reshape(dec_mean_unroll, [-1, original_x_dim])) ** 2, axis=1)
    np_decision = create_label_based_on_quantile(error, quantile=99)

    # TODO metrics computation.

    # %%
    if config.save_figure:
        if original_x_dim == 1:
            plt.figure(figsize=(9, 3))
            plt.plot(x_original_unroll, color='blue', lw=1.5)
            plt.title('Original Data')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./save_figures/{}/Ori_VQRAE_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=600)
            plt.close()

            # Plot decoder output
            plt.figure(figsize=(9, 3))
            plt.plot(dec_mean_unroll, color='blue', lw=1.5)
            plt.title('Decoding Output')
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            plt.savefig('./save_figures/{}/Dec_VQRAE_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=600)
            plt.close()

            plt.figure(figsize=(9, 3))
            plt.plot(error, color='blue', lw=1.5)
            plt.title('Score Output')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                './save_figures/{}/Score_VQRAE_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid),
                dpi=600)
            plt.close()

            t = np.arange(0, abnormal_data.shape[0])
            markercolors = ['blue' if i == 1 else 'red' for i in abnormal_label[config.rolling_size - 1:]]
            markersize = [4 if i == 1 else 25 for i in abnormal_label[config.rolling_size - 1:]]
            plt.figure(figsize=(9, 3))
            ax = plt.axes()
            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()
            plt.margins(0.1)
            plt.scatter(t[: dec_mean_unroll.shape[0]], x_original_unroll, s=markersize, c=markercolors)
            plt.savefig('./save_figures/{}/VisInp_VQRAE_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=600)
            plt.close()

            markercolors = ['blue' if i == 1 else 'red' for i in np_decision]
            markersize = [4 if i == 1 else 25 for i in np_decision]
            plt.figure(figsize=(9, 3))
            ax = plt.axes()
            # plt.yticks([0, 0.25, 0.5, 0.75, 1])
            # ax.set_xlim(t[0] - 10, t[-1] + 10)
            # ax.set_ylim(-0.10, 1.10)
            plt.xlabel('$t$')
            plt.ylabel('$s$')
            plt.grid(True)
            plt.tight_layout()
            plt.margins(0.1)
            # plt.plot(np.squeeze(abnormal_data[config.rolling_size - 1:]), alpha=0.7)
            plt.scatter(t[: np_decision.shape[0]], abnormal_data[config.rolling_size - 1:], s=markersize, c=markercolors)
            # plt.show()
            plt.savefig('./save_figures/{}/VisOut_VQRAE_{}_pid={}.png'.format(config.dataset, Path(file_name).stem, config.pid), dpi=600)
            plt.close()
        else:
            file_logger.info('cannot plot image with x_dim > 1')

    if config.use_spot:
        pass
    else:
        try:
            pos_label = -1
            cm = confusion_matrix(y_true=abnormal_segment, y_pred=np_decision, labels=[1, -1])
            precision = precision_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
            recall = recall_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label)
            fbeta = fbeta_score(y_true=abnormal_segment, y_pred=np_decision, pos_label=pos_label, beta=0.5)
            fpr, tpr, _ = roc_curve(y_true=np.nan_to_num(abnormal_segment), y_score=np.nan_to_num(error), pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            pre, re, _ = precision_recall_curve(y_true=np.nan_to_num(abnormal_segment), probas_pred=np.nan_to_num(error), pos_label=pos_label)
            pr_auc = auc(re, pre)
            cks = cohen_kappa_score(y1=abnormal_segment, y2=np_decision)
            settings = config.to_string()
            insert_sql = """INSERT or REPLACE into model (model_name, pid, settings, dataset, file_name, TN, FP, FN, 
            TP, precision, recall, fbeta, pr_auc, roc_auc, cks, best_TN, best_FP, best_FN, best_TP, best_precision, 
            best_recall, best_fbeta, best_pr_auc, best_roc_auc, best_cks) VALUES('{}', '{}', '{}', '{}', '{}', '{}', 
            '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}', 
            '{}', '{}')""".format(
                'VQRAE', config.pid, settings, config.dataset, Path(file_name).stem, cm[0][0], cm[0][1], cm[1][0],
                cm[1][1], precision, recall, fbeta, pr_auc, roc_auc, cks, vqrae_output.best_TN, vqrae_output.best_FP,
                vqrae_output.best_FN, vqrae_output.best_TP, vqrae_output.best_precision, vqrae_output.best_recall,
                vqrae_output.best_fbeta, vqrae_output.best_pr_auc, vqrae_output.best_roc_auc, vqrae_output.best_cks)
            cursor_obj.execute(insert_sql)
            conn.commit()

            metrics_result = MetricsResult(TN=cm[0][0], FP=cm[0][1], FN=cm[1][0], TP=cm[1][1], precision=precision,
                                           recall=recall, fbeta=fbeta, pr_auc=pr_auc, roc_auc=roc_auc, cks=cks,
                                           best_TN=vqrae_output.best_TN, best_FP=vqrae_output.best_FP,
                                           best_FN=vqrae_output.best_FN, best_TP=vqrae_output.best_TP,
                                           best_precision=vqrae_output.best_precision, best_recall=vqrae_output.best_recall,
                                           best_fbeta=vqrae_output.best_fbeta, best_pr_auc=vqrae_output.best_pr_auc,
                                           best_roc_auc=vqrae_output.best_roc_auc, best_cks=vqrae_output.best_cks,
                                           training_time=0,testing_time=0)
            return metrics_result
        except:
            pass

if __name__ == '__main__':
    conn = sqlite3.connect('./databases/experiments.db')
    cursor_obj = conn.cursor()

    # %%
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-data', type=int, default=0)
    parser.add_argument('--x_dim', '-xd', type=int, default=1)
    parser.add_argument('--h_dim', '-hd', type=int, default=32)
    parser.add_argument('--z_dim', '-zd', type=int, default=16)
    parser.add_argument('--preprocessing', '-pre', type=str2bool, default=True)
    parser.add_argument('--use_overlapping', type=str2bool, default=True)
    parser.add_argument('--rolling_size', '-rs', type=int, default=32)
    parser.add_argument('--epochs', '-ep', type=int, default=50)
    parser.add_argument('--milestone_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--annealing', type=str2bool, default=False)
    parser.add_argument('--early_stopping', type=str2bool, default=False)
    parser.add_argument('--loss_function', type=str, default='beta_1')
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--lmbda', type=float, default=0.0001)
    parser.add_argument('--use_clip_norm', '-uc', type=str2bool, default=True)
    parser.add_argument('--gradient_clip_norm', type=float, default=10)
    parser.add_argument('--use_PNF', '-up', type=str2bool, default=False)
    parser.add_argument('--PNF_layers', type=int, default=10)
    parser.add_argument('--use_bidirection', '-ub', type=str2bool, default=False)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--robust_coeff', type=float, default=0.005)
    parser.add_argument('--display_epoch', '-de', type=int, default=5)
    parser.add_argument('--save_output', type=str2bool, default=True)
    parser.add_argument('--save_figure', type=str2bool, default=False)
    parser.add_argument('--save_model', type=str2bool, default=False)  # save model
    parser.add_argument('--load_model', type=str2bool, default=False)  # load model
    parser.add_argument('--continue_training', type=str2bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use_spot', type=str2bool, default=False)
    parser.add_argument('--use_last_point', '-ulp', type=str2bool, default=True)
    parser.add_argument('--save_config', type=str2bool, default=True)
    parser.add_argument('--load_config', type=str2bool, default=False)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--server_run', type=str2bool, default=False)
    parser.add_argument('--robustness', type=str2bool, default=False)
    parser.add_argument('--pid', type=int, default=0)
    parser.add_argument('--length', type=float, default=1000)
    parser.add_argument('--max', type=float, default=100)
    parser.add_argument('--theta', type=float, default=10)
    parser.add_argument('--anomalies', type=int, default=100)
    args = parser.parse_args()

    if args.load_config:
        config = VQRAEConfig(dataset=None, x_dim=None, h_dim=None, z_dim=None, preprocessing=None, use_overlapping=None,
                             rolling_size=None, epochs=None, milestone_epochs=None, lr=None, gamma=None,
                             batch_size=None, weight_decay=None, annealing=None, early_stopping=None,
                             loss_function=None, rnn_layers=None, lmbda=None, use_clip_norm=None,
                             gradient_clip_norm=None, use_PNF=None, PNF_layers=None, use_bidirection=None,
                             robust_coeff=None, display_epoch=None, save_output=None, save_figure=None,
                             save_model=None, load_model=None, continue_training=None, dropout=None, use_spot=None,
                             use_last_point=None, save_config=None, load_config=None, server_run=None, robustness=None,
                             pid=None)
        try:
            config.import_config('./config/{}/Config_VQRAE_pid={}.json'.format(config.dataset, config.pid))
        except:
            print('There is no config.')
    else:
        config = VQRAEConfig(dataset=args.dataset, x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim,
                             preprocessing=args.preprocessing, use_overlapping=args.use_overlapping,
                             rolling_size=args.rolling_size, epochs=args.epochs, milestone_epochs=args.milestone_epochs,
                             lr=args.lr, gamma=args.gamma, batch_size=args.batch_size, weight_decay=args.weight_decay,
                             annealing=args.annealing, early_stopping=args.early_stopping,
                             loss_function=args.loss_function, lmbda=args.lmbda, use_clip_norm=args.use_clip_norm,
                             gradient_clip_norm=args.gradient_clip_norm, rnn_layers=args.rnn_layers,
                             use_PNF=args.use_PNF, PNF_layers=args.PNF_layers, use_bidirection=args.use_bidirection,
                             robust_coeff=args.robust_coeff, display_epoch=args.display_epoch,
                             save_output=args.save_output, save_figure=args.save_figure, save_model=args.save_model,
                             load_model=args.load_model, continue_training=args.continue_training, dropout=args.dropout,
                             use_spot=args.use_spot, use_last_point=args.use_last_point, save_config=args.save_config,
                             load_config=args.load_config, server_run=args.server_run, robustness=args.robustness,
                             pid=args.pid)
    if args.save_config:
        if not os.path.exists('./save_configs/{}/'.format(args.dataset)):
            os.makedirs('./save_configs/{}/'.format(args.dataset))
        config.export_config('./save_configs/{}/Config_VQRAE_pid={}.json'.format(args.dataset, args.pid))

    if args.save_model:
        if not os.path.exists('./save_models/{}/'.format(args.dataset)):
            os.makedirs('./save_models/{}/'.format(args.dataset))

    if args.save_output:
        if not os.path.exists('./save_outputs/NPY/{}/'.format(args.dataset)):
            os.makedirs('./save_outputs/NPY/{}/'.format(args.dataset))


    if args.save_figure:
        if not os.path.exists('./save_figures/{}/'.format(args.dataset)):
            os.makedirs('./save_figures/{}/'.format(args.dataset))

    # %%
    device = torch.device(get_free_device())

    train_logger, file_logger, meta_logger = create_logger(dataset=args.dataset,
                                                           train_logger_name='vqrae_train_logger',
                                                           file_logger_name='vqrae_file_logger',
                                                           meta_logger_name='vqrae_meta_logger',
                                                           model_name='VQRAE',
                                                           pid=args.pid)

    # logging setting
    file_logger.info('============================')
    for key, value in vars(args).items():
        file_logger.info(key + ' = {}'.format(value))
    file_logger.info('============================')

    meta_logger.info('============================')
    for key, value in vars(args).items():
        meta_logger.info(key + ' = {}'.format(value))
    meta_logger.info('============================')

    path = None
    paths = None

    start_time_file = time.time()
    execution_time = 0
    
    if args.dataset == 0:
        file_name = 'synthetic'
        file_logger.info(file_name)
        if args.server_run:
            try:
                metrics_result = RunModel(file_name=file_name, config=config)
                meta_logger.info(file_name)
                meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
                meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
                meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
                meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
                meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
                meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
                meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
                meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
                meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
                meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
                meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
                meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
                meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
                meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
                meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
                meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
                meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
                meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
                meta_logger.info('training_time = {}'.format(metrics_result.training_time))
                meta_logger.info('testing_time = {}'.format(metrics_result.testing_time))
                meta_logger.info('memory_estimation = {}'.format(metrics_result.memory_estimation))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
            except Exception as e:
                file_logger.info(str(traceback.format_exc()) + str(e))
        else:
            metrics_result = RunModel(file_name=file_name, config=config)
            meta_logger.info(file_name)
            meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
            meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
            meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
            meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
            meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
            meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
            meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
            meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
            meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
            meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
            meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
            meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
            meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
            meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
            meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
            meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
            meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
            meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
            meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
            meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
            meta_logger.info('training_time = {}'.format(metrics_result.training_time))
            meta_logger.info('testing_time = {}'.format(metrics_result.testing_time))
            meta_logger.info('memory_estimation = {}'.format(metrics_result.memory_estimation))
            file_logger.info('Finish')
            # logger.shutdown()
            meta_logger.info('Finish')

    if args.dataset == 1:
        file_name = './data/GD/data/Genesis_AnomalyLabels.csv'
        file_logger.info(file_name)
        if args.server_run:
            try:
                metrics_result = RunModel(file_name=file_name, config=config)
                meta_logger.info(file_name)
                meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
                meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
                meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
                meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
                meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
                meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
                meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
                meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
                meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
                meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
                meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
                meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
                meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
                meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
                meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
                meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
                meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
                meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
                meta_logger.info('training_time = {}'.format(metrics_result.training_time))
                meta_logger.info('testing_time = {}'.format(metrics_result.testing_time))
                meta_logger.info('memory_estimation = {}'.format(metrics_result.memory_estimation))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
            except Exception as e:
                file_logger.info(str(traceback.format_exc()) + str(e))
        else:
            metrics_result = RunModel(file_name=file_name, config=config)
            meta_logger.info(file_name)
            meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
            meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
            meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
            meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
            meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
            meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
            meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
            meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
            meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
            meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
            meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
            meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
            meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
            meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
            meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
            meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
            meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
            meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
            meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
            meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
            meta_logger.info('training_time = {}'.format(metrics_result.training_time))
            meta_logger.info('testing_time = {}'.format(metrics_result.testing_time))
            meta_logger.info('memory_estimation = {}'.format(metrics_result.memory_estimation))
            file_logger.info('Finish')
            # logger.shutdown()
            meta_logger.info('Finish')

    if args.dataset == 2:
        file_name = './data/HSS/data/HRSS_anomalous_standard.csv'
        file_logger.info(file_name)
        if args.server_run:
            try:
                metrics_result = RunModel(file_name=file_name, config=config)
                meta_logger.info(file_name)
                meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
                meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
                meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
                meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
                meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
                meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
                meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
                meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
                meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
                meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
                meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
                meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
                meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
                meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
                meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
                meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
                meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
                meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
                meta_logger.info('training_time = {}'.format(metrics_result.training_time))
                meta_logger.info('testing_time = {}'.format(metrics_result.testing_time))
                meta_logger.info('memory_estimation = {}'.format(metrics_result.memory_estimation))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
            except Exception as e:
                file_logger.info(str(traceback.format_exc()) + str(e))
        else:
            metrics_result = RunModel(file_name=file_name, config=config)
            meta_logger.info(file_name)
            meta_logger.info('avg_TN = {}'.format(metrics_result.TN))
            meta_logger.info('avg_FP = {}'.format(metrics_result.FP))
            meta_logger.info('avg_FN = {}'.format(metrics_result.FN))
            meta_logger.info('avg_TP = {}'.format(metrics_result.TP))
            meta_logger.info('avg_precision = {}'.format(metrics_result.precision))
            meta_logger.info('avg_recall = {}'.format(metrics_result.recall))
            meta_logger.info('avg_fbeta = {}'.format(metrics_result.fbeta))
            meta_logger.info('avg_roc_auc = {}'.format(metrics_result.roc_auc))
            meta_logger.info('avg_pr_auc = {}'.format(metrics_result.pr_auc))
            meta_logger.info('avg_cks = {}'.format(metrics_result.cks))
            meta_logger.info('avg_best_TN = {}'.format(metrics_result.best_TN))
            meta_logger.info('avg_best_FP = {}'.format(metrics_result.best_FP))
            meta_logger.info('avg_best_FN = {}'.format(metrics_result.best_FN))
            meta_logger.info('avg_best_TP = {}'.format(metrics_result.best_TP))
            meta_logger.info('avg_best_precision = {}'.format(metrics_result.best_precision))
            meta_logger.info('avg_best_recall = {}'.format(metrics_result.best_recall))
            meta_logger.info('avg_best_fbeta = {}'.format(metrics_result.best_fbeta))
            meta_logger.info('avg_best_roc_auc = {}'.format(metrics_result.best_roc_auc))
            meta_logger.info('avg_best_pr_auc = {}'.format(metrics_result.best_pr_auc))
            meta_logger.info('avg_best_cks = {}'.format(metrics_result.best_cks))
            meta_logger.info('training_time = {}'.format(metrics_result.training_time))
            meta_logger.info('testing_time = {}'.format(metrics_result.testing_time))
            meta_logger.info('memory_estimation = {}'.format(metrics_result.memory_estimation))
            file_logger.info('Finish')
            # logger.shutdown()
            meta_logger.info('Finish')

    if args.dataset == 31:
        path = './data/YAHOO/data/A1Benchmark'
    if args.dataset == 32:
        path = './data/YAHOO/data/A2Benchmark'
    if args.dataset == 33:
        path = './data/YAHOO/data/A3Benchmark'
    if args.dataset == 34:
        path = './data/YAHOO/data/A4Benchmark'
    if args.dataset == 35:
        path = './data/YAHOO/data/Vis'
    if args.dataset == 41:
        path = './data/NAB/data/artificialWithAnomaly'
    if args.dataset == 42:
        path = './data/NAB/data/realAdExchange'
    if args.dataset == 43:
        path = './data/NAB/data/realAWSCloudwatch'
    if args.dataset == 44:
        path = './data/NAB/data/realKnownCause'
    if args.dataset == 45:
        path = './data/NAB/data/realTraffic'
    if args.dataset == 46:
        path = './data/NAB/data/realTweets'
    if args.dataset == 51:
        path = './data/2D/Comb/test'
    if args.dataset == 52:
        path = './data/2D/Cross/test'
    if args.dataset == 53:
        path = './data/2D/Intersection/test'
    if args.dataset == 54:
        path = './data/2D/Pentagram/test'
    if args.dataset == 55:
        path = './data/2D/Ring/test'
    if args.dataset == 56:
        path = './data/2D/Stripe/test'
    if args.dataset == 57:
        path = './data/2D/Triangle/test'
    if args.dataset == 61:
        path = './data/ECG/chf01'
    if args.dataset == 62:
        path = './data/ECG/chf13'
    if args.dataset == 63:
        path = './data/ECG/ltstdb43'
    if args.dataset == 64:
        path = './data/ECG/ltstdb240'
    if args.dataset == 65:
        path = './data/ECG/mitdb180'
    if args.dataset == 66:
        path = './data/ECG/stdb308'
    if args.dataset == 67:
        path = './data/ECG/xmitdb108'
    if args.dataset == 71:
        path = './data/SMD/machine1/train'
    if args.dataset == 72:
        path = './data/SMD/machine2/train'
    if args.dataset == 73:
        path = './data/SMD/machine3/train'
    if args.dataset == 81:
        path = './data/SMAP/channel1/train'
    if args.dataset == 82:
        path = './data/SMAP/channel2/train'
    if args.dataset == 83:
        path = './data/SMAP/channel3/train'
    if args.dataset == 84:
        path = './data/SMAP/channel4/train'
    if args.dataset == 85:
        path = './data/SMAP/channel5/train'
    if args.dataset == 86:
        path = './data/SMAP/channel6/train'
    if args.dataset == 87:
        path = './data/SMAP/channel7/train'
    if args.dataset == 88:
        path = './data/SMAP/channel8/train'
    if args.dataset == 89:
        path = './data/SMAP/channel9/train'
    if args.dataset == 90:
        path = './data/SMAP/channel10/train'
    if args.dataset == 91:
        path = './data/MSL/channel1/train'
    if args.dataset == 92:
        path = './data/MSL/channel2/train'
    if args.dataset == 93:
        path = './data/MSL/channel3/train'
    if args.dataset == 94:
        path = './data/MSL/channel4/train'
    if args.dataset == 95:
        path = './data/MSL/channel5/train'
    if args.dataset == 96:
        path = './data/MSL/channel6/train'
    if args.dataset == 97:
        path = './data/MSL/channel7/train'
    if args.dataset == 101:
        path = './data/SWaT/train'
    if args.dataset == 111:
        path = './data/WADI/2017/train'
    if args.dataset == 112:
        path = './data/WADI/2019/train'

    if args.dataset == 3:
        paths = ['./data/YAHOO/data/A1Benchmark', './data/YAHOO/data/A2Benchmark', './data/YAHOO/data/A3Benchmark', './data/YAHOO/data/A4Benchmark']
    if args.dataset == 4:
        paths = ['./data/NAB/data/artificialWithAnomaly', './data/NAB/data/realAdExchange', './data/NAB/data/realAWSCloudwatch', './data/NAB/data/realKnownCause', './data/NAB/data/realTraffic', './data/NAB/data/realTweets']
    if args.dataset == 5:
        paths = ['./data/2D/Comb', './data/2D/Cross', './data/2D/Intersection', './data/2D/Pentagram', './data/2D/Ring', './data/2D/Stripe', './data/2D/Triangle']
    if args.dataset == 6:
        paths = ['./data/ECG/chf01', './data/ECG/chf13', './data/ECG/ltstdb43', './data/ECG/ltstdb240', './data/ECG/mitdb180', './data/ECG/stdb308', './data/ECG/xmitdb108']
    if args.dataset == 7:
        paths = ['./data/SMD/machine1/train', './data/SMD/machine2/train', './data/SMD/machine3/train']
    if args.dataset == 8:
        paths = ['./data/SMAP/channel1/train', './data/SMAP/channel2/train', './data/SMAP/channel3/train', './data/SMAP/channel4/train', './data/SMAP/channel5/train', './data/SMAP/channel6/train', './data/SMAP/channel7/train', './data/SMAP/channel8/train', './data/SMAP/channel9/train', './data/SMAP/channel10/train']
    if args.dataset == 9:
        paths = ['./data/MSL/channel1/train', './data/MSL/channel2/train', './data/MSL/channel3/train', './data/MSL/channel4/train', './data/MSL/channel5/train', './data/MSL/channel6/train', './data/MSL/channel7/train']
    if args.dataset == 11:
        paths = ['./data/WADI/2017/train', './data/WADI/2019/train']

    if paths is not None:
        for path in paths:
            for root, dirs, files in os.walk(path):
                if len(dirs) == 0:
                    s_TN = []
                    s_FP = []
                    s_FN = []
                    s_TP = []
                    s_precision = []
                    s_recall = []
                    s_fbeta = []
                    s_roc_auc = []
                    s_pr_auc = []
                    s_cks = []
                    s_best_TN = []
                    s_best_FP = []
                    s_best_FN = []
                    s_best_TP = []
                    s_best_precision = []
                    s_best_recall = []
                    s_best_fbeta = []
                    s_best_roc_auc = []
                    s_best_pr_auc = []
                    s_best_cks = []
                    s_training_time = []
                    s_testing_time = []
                    s_memory_estimation = []
                    for file in files:
                        file_name = os.path.join(root, file)
                        file_logger.info('============================')
                        file_logger.info(file)

                        if args.server_run:
                            try:
                                metrics_result = RunModel(file_name=file_name, config=config)
                                s_TN.append(metrics_result.TN)
                                file_logger.info('TN = {}'.format(metrics_result.TN))
                                s_FP.append(metrics_result.FP)
                                file_logger.info('FP = {}'.format(metrics_result.FP))
                                s_FN.append(metrics_result.FN)
                                file_logger.info('FN = {}'.format(metrics_result.FN))
                                s_TP.append(metrics_result.TP)
                                file_logger.info('TP = {}'.format(metrics_result.TP))
                                s_precision.append(metrics_result.precision)
                                file_logger.info('precision = {}'.format(metrics_result.precision))
                                s_recall.append(metrics_result.recall)
                                file_logger.info('recall = {}'.format(metrics_result.recall))
                                s_fbeta.append(metrics_result.fbeta)
                                file_logger.info('fbeta = {}'.format(metrics_result.fbeta))
                                s_roc_auc.append(metrics_result.roc_auc)
                                file_logger.info('roc_auc = {}'.format(metrics_result.roc_auc))
                                s_pr_auc.append(metrics_result.pr_auc)
                                file_logger.info('pr_auc = {}'.format(metrics_result.pr_auc))
                                s_cks.append(metrics_result.cks)
                                file_logger.info('cks = {}'.format(metrics_result.cks))
                                s_best_TN.append(metrics_result.best_TN)
                                file_logger.info('best_TN = {}'.format(metrics_result.best_TN))
                                s_best_FP.append(metrics_result.best_FP)
                                file_logger.info('best_FP = {}'.format(metrics_result.best_FP))
                                s_best_FN.append(metrics_result.best_FN)
                                file_logger.info('best_FN = {}'.format(metrics_result.best_FN))
                                s_best_TP.append(metrics_result.best_TP)
                                file_logger.info('best_TP = {}'.format(metrics_result.best_TP))
                                s_best_precision.append(metrics_result.best_precision)
                                file_logger.info('best_precision = {}'.format(metrics_result.best_precision))
                                s_best_recall.append(metrics_result.best_recall)
                                file_logger.info('best_recall = {}'.format(metrics_result.best_recall))
                                s_best_fbeta.append(metrics_result.best_fbeta)
                                file_logger.info('best_fbeta = {}'.format(metrics_result.best_fbeta))
                                s_best_roc_auc.append(metrics_result.best_roc_auc)
                                file_logger.info('best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                                s_best_pr_auc.append(metrics_result.best_pr_auc)
                                file_logger.info('best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                                s_best_cks.append(metrics_result.best_cks)
                                file_logger.info('best_cks = {}'.format(metrics_result.best_cks))
                                s_training_time.append(metrics_result.training_time)
                                file_logger.info('training_time = {}'.format(metrics_result.training_time))
                                s_testing_time.append(metrics_result.testing_time)
                                file_logger.info('testing_time = {}'.format(metrics_result.testing_time))
                                s_memory_estimation.append(metrics_result.memory_estimation)
                                file_logger.info('memory_estimation = {}'.format(metrics_result.memory_estimation))
                            except Exception as e:
                                file_logger.info(str(traceback.format_exc()) + str(e))
                                continue
                        else:
                            metrics_result = RunModel(file_name=file_name, config=config)
                            s_TN.append(metrics_result.TN)
                            file_logger.info('TN = {}'.format(metrics_result.TN))
                            s_FP.append(metrics_result.FP)
                            file_logger.info('FP = {}'.format(metrics_result.FP))
                            s_FN.append(metrics_result.FN)
                            file_logger.info('FN = {}'.format(metrics_result.FN))
                            s_TP.append(metrics_result.TP)
                            file_logger.info('TP = {}'.format(metrics_result.TP))
                            s_precision.append(metrics_result.precision)
                            file_logger.info('precision = {}'.format(metrics_result.precision))
                            s_recall.append(metrics_result.recall)
                            file_logger.info('recall = {}'.format(metrics_result.recall))
                            s_fbeta.append(metrics_result.fbeta)
                            file_logger.info('fbeta = {}'.format(metrics_result.fbeta))
                            s_roc_auc.append(metrics_result.roc_auc)
                            file_logger.info('roc_auc = {}'.format(metrics_result.roc_auc))
                            s_pr_auc.append(metrics_result.pr_auc)
                            file_logger.info('pr_auc = {}'.format(metrics_result.pr_auc))
                            s_cks.append(metrics_result.cks)
                            file_logger.info('cks = {}'.format(metrics_result.cks))
                            s_best_TN.append(metrics_result.best_TN)
                            file_logger.info('best_TN = {}'.format(metrics_result.best_TN))
                            s_best_FP.append(metrics_result.best_FP)
                            file_logger.info('best_FP = {}'.format(metrics_result.best_FP))
                            s_best_FN.append(metrics_result.best_FN)
                            file_logger.info('best_FN = {}'.format(metrics_result.best_FN))
                            s_best_TP.append(metrics_result.best_TP)
                            file_logger.info('best_TP = {}'.format(metrics_result.best_TP))
                            s_best_precision.append(metrics_result.best_precision)
                            file_logger.info('best_precision = {}'.format(metrics_result.best_precision))
                            s_best_recall.append(metrics_result.best_recall)
                            file_logger.info('best_recall = {}'.format(metrics_result.best_recall))
                            s_best_fbeta.append(metrics_result.best_fbeta)
                            file_logger.info('best_fbeta = {}'.format(metrics_result.best_fbeta))
                            s_best_roc_auc.append(metrics_result.best_roc_auc)
                            file_logger.info('best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                            s_best_pr_auc.append(metrics_result.best_pr_auc)
                            file_logger.info('best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                            s_best_cks.append(metrics_result.best_cks)
                            file_logger.info('best_cks = {}'.format(metrics_result.best_cks))
                            s_training_time.append(metrics_result.training_time)
                            file_logger.info('training_time = {}'.format(metrics_result.training_time))
                            s_testing_time.append(metrics_result.testing_time)
                            file_logger.info('testing_time = {}'.format(metrics_result.testing_time))
                            s_memory_estimation.append(metrics_result.memory_estimation)
                            file_logger.info('memory_estimation = {}'.format(metrics_result.memory_estimation))
                meta_logger.info(dir)
                avg_TN = calculate_average_metric(s_TN)
                meta_logger.info('avg_TN = {}'.format(avg_TN))
                avg_FP = calculate_average_metric(s_FP)
                meta_logger.info('avg_FP = {}'.format(avg_FP))
                avg_FN = calculate_average_metric(s_FN)
                meta_logger.info('avg_FN = {}'.format(avg_FN))
                avg_TP = calculate_average_metric(s_TP)
                meta_logger.info('avg_TP = {}'.format(avg_TP))
                avg_precision = calculate_average_metric(s_precision)
                meta_logger.info('avg_precision = {}'.format(avg_precision))
                avg_recall = calculate_average_metric(s_recall)
                meta_logger.info('avg_recall = {}'.format(avg_recall))
                avg_fbeta = calculate_average_metric(s_fbeta)
                meta_logger.info('avg_fbeta = {}'.format(avg_fbeta))
                avg_roc_auc = calculate_average_metric(s_roc_auc)
                meta_logger.info('avg_roc_auc = {}'.format(avg_roc_auc))
                avg_pr_auc = calculate_average_metric(s_pr_auc)
                meta_logger.info('avg_pr_auc = {}'.format(avg_pr_auc))
                avg_cks = calculate_average_metric(s_cks)
                meta_logger.info('avg_cks = {}'.format(avg_cks))
                avg_best_TN = calculate_average_metric(s_best_TN)
                meta_logger.info('avg_best_TN = {}'.format(avg_best_TN))
                avg_best_FP = calculate_average_metric(s_best_FP)
                meta_logger.info('avg_best_FP = {}'.format(avg_best_FP))
                avg_best_FN = calculate_average_metric(s_best_FN)
                meta_logger.info('avg_best_FN = {}'.format(avg_best_FN))
                avg_best_TP = calculate_average_metric(s_best_TP)
                meta_logger.info('avg_best_TP = {}'.format(avg_best_TP))
                avg_best_precision = calculate_average_metric(s_best_precision)
                meta_logger.info('avg_best_precision = {}'.format(avg_best_precision))
                avg_best_recall = calculate_average_metric(s_best_recall)
                meta_logger.info('avg_best_recall = {}'.format(avg_best_recall))
                avg_best_fbeta = calculate_average_metric(s_best_fbeta)
                meta_logger.info('avg_best_fbeta = {}'.format(avg_best_fbeta))
                avg_best_roc_auc = calculate_average_metric(s_best_roc_auc)
                meta_logger.info('avg_best_roc_auc = {}'.format(avg_best_roc_auc))
                avg_best_pr_auc = calculate_average_metric(s_best_pr_auc)
                meta_logger.info('avg_best_pr_auc = {}'.format(avg_best_pr_auc))
                avg_best_cks = calculate_average_metric(s_best_cks)
                meta_logger.info('avg_best_cks = {}'.format(avg_best_cks))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
                # meta_logger.shutdown()
        path = None
    if path is not None:
        for root, dirs, files in os.walk(path):
            if len(dirs) == 0:
                s_TN = []
                s_FP = []
                s_FN = []
                s_TP = []
                s_precision = []
                s_recall = []
                s_fbeta = []
                s_roc_auc = []
                s_pr_auc = []
                s_cks = []
                s_best_TN = []
                s_best_FP = []
                s_best_FN = []
                s_best_TP = []
                s_best_precision = []
                s_best_recall = []
                s_best_fbeta = []
                s_best_roc_auc = []
                s_best_pr_auc = []
                s_best_cks = []
                s_training_time = []
                s_testing_time = []
                s_memory_estimation = []
                for file in files:
                    file_name = os.path.join(root, file)
                    file_logger.info('============================')
                    file_logger.info(file)

                    if args.server_run:
                        try:
                            metrics_result = RunModel(file_name=file_name, config=config)
                            s_TN.append(metrics_result.TN)
                            file_logger.info('TN = {}'.format(metrics_result.TN))
                            s_FP.append(metrics_result.FP)
                            file_logger.info('FP = {}'.format(metrics_result.FP))
                            s_FN.append(metrics_result.FN)
                            file_logger.info('FN = {}'.format(metrics_result.FN))
                            s_TP.append(metrics_result.TP)
                            file_logger.info('TP = {}'.format(metrics_result.TP))
                            s_precision.append(metrics_result.precision)
                            file_logger.info('precision = {}'.format(metrics_result.precision))
                            s_recall.append(metrics_result.recall)
                            file_logger.info('recall = {}'.format(metrics_result.recall))
                            s_fbeta.append(metrics_result.fbeta)
                            file_logger.info('fbeta = {}'.format(metrics_result.fbeta))
                            s_roc_auc.append(metrics_result.roc_auc)
                            file_logger.info('roc_auc = {}'.format(metrics_result.roc_auc))
                            s_pr_auc.append(metrics_result.pr_auc)
                            file_logger.info('pr_auc = {}'.format(metrics_result.pr_auc))
                            s_cks.append(metrics_result.cks)
                            file_logger.info('cks = {}'.format(metrics_result.cks))
                            s_best_TN.append(metrics_result.best_TN)
                            file_logger.info('best_TN = {}'.format(metrics_result.best_TN))
                            s_best_FP.append(metrics_result.best_FP)
                            file_logger.info('best_FP = {}'.format(metrics_result.best_FP))
                            s_best_FN.append(metrics_result.best_FN)
                            file_logger.info('best_FN = {}'.format(metrics_result.best_FN))
                            s_best_TP.append(metrics_result.best_TP)
                            file_logger.info('best_TP = {}'.format(metrics_result.best_TP))
                            s_best_precision.append(metrics_result.best_precision)
                            file_logger.info('best_precision = {}'.format(metrics_result.best_precision))
                            s_best_recall.append(metrics_result.best_recall)
                            file_logger.info('best_recall = {}'.format(metrics_result.best_recall))
                            s_best_fbeta.append(metrics_result.best_fbeta)
                            file_logger.info('best_fbeta = {}'.format(metrics_result.best_fbeta))
                            s_best_roc_auc.append(metrics_result.best_roc_auc)
                            file_logger.info('best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                            s_best_pr_auc.append(metrics_result.best_pr_auc)
                            file_logger.info('best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                            s_best_cks.append(metrics_result.best_cks)
                            file_logger.info('best_cks = {}'.format(metrics_result.best_cks))
                        except Exception as e:
                            file_logger.info(str(traceback.format_exc()) + str(e))
                            continue
                    else:
                        metrics_result = RunModel(file_name=file_name, config=config)
                        s_TN.append(metrics_result.TN)
                        file_logger.info('TN = {}'.format(metrics_result.TN))
                        s_FP.append(metrics_result.FP)
                        file_logger.info('FP = {}'.format(metrics_result.FP))
                        s_FN.append(metrics_result.FN)
                        file_logger.info('FN = {}'.format(metrics_result.FN))
                        s_TP.append(metrics_result.TP)
                        file_logger.info('TP = {}'.format(metrics_result.TP))
                        s_precision.append(metrics_result.precision)
                        file_logger.info('precision = {}'.format(metrics_result.precision))
                        s_recall.append(metrics_result.recall)
                        file_logger.info('recall = {}'.format(metrics_result.recall))
                        s_fbeta.append(metrics_result.fbeta)
                        file_logger.info('fbeta = {}'.format(metrics_result.fbeta))
                        s_roc_auc.append(metrics_result.roc_auc)
                        file_logger.info('roc_auc = {}'.format(metrics_result.roc_auc))
                        s_pr_auc.append(metrics_result.pr_auc)
                        file_logger.info('pr_auc = {}'.format(metrics_result.pr_auc))
                        s_cks.append(metrics_result.cks)
                        file_logger.info('cks = {}'.format(metrics_result.cks))
                        s_best_TN.append(metrics_result.best_TN)
                        file_logger.info('best_TN = {}'.format(metrics_result.best_TN))
                        s_best_FP.append(metrics_result.best_FP)
                        file_logger.info('best_FP = {}'.format(metrics_result.best_FP))
                        s_best_FN.append(metrics_result.best_FN)
                        file_logger.info('best_FN = {}'.format(metrics_result.best_FN))
                        s_best_TP.append(metrics_result.best_TP)
                        file_logger.info('best_TP = {}'.format(metrics_result.best_TP))
                        s_best_precision.append(metrics_result.best_precision)
                        file_logger.info('best_precision = {}'.format(metrics_result.best_precision))
                        s_best_recall.append(metrics_result.best_recall)
                        file_logger.info('best_recall = {}'.format(metrics_result.best_recall))
                        s_best_fbeta.append(metrics_result.best_fbeta)
                        file_logger.info('best_fbeta = {}'.format(metrics_result.best_fbeta))
                        s_best_roc_auc.append(metrics_result.best_roc_auc)
                        file_logger.info('best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                        s_best_pr_auc.append(metrics_result.best_pr_auc)
                        file_logger.info('best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                        s_best_cks.append(metrics_result.best_cks)
                        file_logger.info('best_cks = {}'.format(metrics_result.best_cks))
                        s_training_time.append(metrics_result.training_time)
                        file_logger.info('training_time = {}'.format(metrics_result.training_time))
                        s_testing_time.append(metrics_result.testing_time)
                        file_logger.info('testing_time = {}'.format(metrics_result.testing_time))
                        s_memory_estimation.append(metrics_result.memory_estimation)
                        file_logger.info('memory_estimation = {}'.format(metrics_result.memory_estimation))
                meta_logger.info(dir)
                avg_TN = calculate_average_metric(s_TN)
                meta_logger.info('avg_TN = {}'.format(avg_TN))
                avg_FP = calculate_average_metric(s_FP)
                meta_logger.info('avg_FP = {}'.format(avg_FP))
                avg_FN = calculate_average_metric(s_FN)
                meta_logger.info('avg_FN = {}'.format(avg_FN))
                avg_TP = calculate_average_metric(s_TP)
                meta_logger.info('avg_TP = {}'.format(avg_TP))
                avg_precision = calculate_average_metric(s_precision)
                meta_logger.info('avg_precision = {}'.format(avg_precision))
                avg_recall = calculate_average_metric(s_recall)
                meta_logger.info('avg_recall = {}'.format(avg_recall))
                avg_fbeta = calculate_average_metric(s_fbeta)
                meta_logger.info('avg_fbeta = {}'.format(avg_fbeta))
                avg_roc_auc = calculate_average_metric(s_roc_auc)
                meta_logger.info('avg_roc_auc = {}'.format(avg_roc_auc))
                avg_pr_auc = calculate_average_metric(s_pr_auc)
                meta_logger.info('avg_pr_auc = {}'.format(avg_pr_auc))
                avg_cks = calculate_average_metric(s_cks)
                meta_logger.info('avg_cks = {}'.format(avg_cks))
                avg_best_TN = calculate_average_metric(s_best_TN)
                meta_logger.info('avg_best_TN = {}'.format(avg_best_TN))
                avg_best_FP = calculate_average_metric(s_best_FP)
                meta_logger.info('avg_best_FP = {}'.format(avg_best_FP))
                avg_best_FN = calculate_average_metric(s_best_FN)
                meta_logger.info('avg_best_FN = {}'.format(avg_best_FN))
                avg_best_TP = calculate_average_metric(s_best_TP)
                meta_logger.info('avg_best_TP = {}'.format(avg_best_TP))
                avg_best_precision = calculate_average_metric(s_best_precision)
                meta_logger.info('avg_best_precision = {}'.format(avg_best_precision))
                avg_best_recall = calculate_average_metric(s_best_recall)
                meta_logger.info('avg_best_recall = {}'.format(avg_best_recall))
                avg_best_fbeta = calculate_average_metric(s_best_fbeta)
                meta_logger.info('avg_best_fbeta = {}'.format(avg_best_fbeta))
                avg_best_roc_auc = calculate_average_metric(s_best_roc_auc)
                meta_logger.info('avg_best_roc_auc = {}'.format(avg_best_roc_auc))
                avg_best_pr_auc = calculate_average_metric(s_best_pr_auc)
                meta_logger.info('avg_best_pr_auc = {}'.format(avg_best_pr_auc))
                avg_best_cks = calculate_average_metric(s_best_cks)
                meta_logger.info('avg_best_cks = {}'.format(avg_best_cks))
                file_logger.info('Finish')
                # logger.shutdown()
                meta_logger.info('Finish')
                # meta_logger.shutdown()
        path = None
    if path is not None:
        for root, dirs, files in os.walk(path):
            if len(dirs) == 0:
                s_TN = []
                s_FP = []
                s_FN = []
                s_TP = []
                s_precision = []
                s_recall = []
                s_fbeta = []
                s_roc_auc = []
                s_pr_auc = []
                s_cks = []
                s_best_TN = []
                s_best_FP = []
                s_best_FN = []
                s_best_TP = []
                s_best_precision = []
                s_best_recall = []
                s_best_fbeta = []
                s_best_roc_auc = []
                s_best_pr_auc = []
                s_best_cks = []
                s_training_time = []
                s_testing_time = []
                s_memory_estimation = []
                for file in files:
                    file_name = os.path.join(root, file)
                    file_logger.info('============================')
                    file_logger.info(file)

                    if args.server_run:
                        try:
                            metrics_result = RunModel(file_name=file_name, config=config)
                            s_TN.append(metrics_result.TN)
                            file_logger.info('TN = {}'.format(metrics_result.TN))
                            s_FP.append(metrics_result.FP)
                            file_logger.info('FP = {}'.format(metrics_result.FP))
                            s_FN.append(metrics_result.FN)
                            file_logger.info('FN = {}'.format(metrics_result.FN))
                            s_TP.append(metrics_result.TP)
                            file_logger.info('TP = {}'.format(metrics_result.TP))
                            s_precision.append(metrics_result.precision)
                            file_logger.info('precision = {}'.format(metrics_result.precision))
                            s_recall.append(metrics_result.recall)
                            file_logger.info('recall = {}'.format(metrics_result.recall))
                            s_fbeta.append(metrics_result.fbeta)
                            file_logger.info('fbeta = {}'.format(metrics_result.fbeta))
                            s_roc_auc.append(metrics_result.roc_auc)
                            file_logger.info('roc_auc = {}'.format(metrics_result.roc_auc))
                            s_pr_auc.append(metrics_result.pr_auc)
                            file_logger.info('pr_auc = {}'.format(metrics_result.pr_auc))
                            s_cks.append(metrics_result.cks)
                            file_logger.info('cks = {}'.format(metrics_result.cks))
                            s_best_TN.append(metrics_result.best_TN)
                            file_logger.info('best_TN = {}'.format(metrics_result.best_TN))
                            s_best_FP.append(metrics_result.best_FP)
                            file_logger.info('best_FP = {}'.format(metrics_result.best_FP))
                            s_best_FN.append(metrics_result.best_FN)
                            file_logger.info('best_FN = {}'.format(metrics_result.best_FN))
                            s_best_TP.append(metrics_result.best_TP)
                            file_logger.info('best_TP = {}'.format(metrics_result.best_TP))
                            s_best_precision.append(metrics_result.best_precision)
                            file_logger.info('best_precision = {}'.format(metrics_result.best_precision))
                            s_best_recall.append(metrics_result.best_recall)
                            file_logger.info('best_recall = {}'.format(metrics_result.best_recall))
                            s_best_fbeta.append(metrics_result.best_fbeta)
                            file_logger.info('best_fbeta = {}'.format(metrics_result.best_fbeta))
                            s_best_roc_auc.append(metrics_result.best_roc_auc)
                            file_logger.info('best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                            s_best_pr_auc.append(metrics_result.best_pr_auc)
                            file_logger.info('best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                            s_best_cks.append(metrics_result.best_cks)
                            file_logger.info('best_cks = {}'.format(metrics_result.best_cks))
                        except Exception as e:
                            file_logger.info(str(traceback.format_exc()) + str(e))
                            continue
                else:
                    metrics_result = RunModel(file_name=file_name, config=config)
                    s_TN.append(metrics_result.TN)
                    file_logger.info('TN = {}'.format(metrics_result.TN))
                    s_FP.append(metrics_result.FP)
                    file_logger.info('FP = {}'.format(metrics_result.FP))
                    s_FN.append(metrics_result.FN)
                    file_logger.info('FN = {}'.format(metrics_result.FN))
                    s_TP.append(metrics_result.TP)
                    file_logger.info('TP = {}'.format(metrics_result.TP))
                    s_precision.append(metrics_result.precision)
                    file_logger.info('precision = {}'.format(metrics_result.precision))
                    s_recall.append(metrics_result.recall)
                    file_logger.info('recall = {}'.format(metrics_result.recall))
                    s_fbeta.append(metrics_result.fbeta)
                    file_logger.info('fbeta = {}'.format(metrics_result.fbeta))
                    s_roc_auc.append(metrics_result.roc_auc)
                    file_logger.info('roc_auc = {}'.format(metrics_result.roc_auc))
                    s_pr_auc.append(metrics_result.pr_auc)
                    file_logger.info('pr_auc = {}'.format(metrics_result.pr_auc))
                    s_cks.append(metrics_result.cks)
                    file_logger.info('cks = {}'.format(metrics_result.cks))
                    s_best_TN.append(metrics_result.best_TN)
                    file_logger.info('best_TN = {}'.format(metrics_result.best_TN))
                    s_best_FP.append(metrics_result.best_FP)
                    file_logger.info('best_FP = {}'.format(metrics_result.best_FP))
                    s_best_FN.append(metrics_result.best_FN)
                    file_logger.info('best_FN = {}'.format(metrics_result.best_FN))
                    s_best_TP.append(metrics_result.best_TP)
                    file_logger.info('best_TP = {}'.format(metrics_result.best_TP))
                    s_best_precision.append(metrics_result.best_precision)
                    file_logger.info('best_precision = {}'.format(metrics_result.best_precision))
                    s_best_recall.append(metrics_result.best_recall)
                    file_logger.info('best_recall = {}'.format(metrics_result.best_recall))
                    s_best_fbeta.append(metrics_result.best_fbeta)
                    file_logger.info('best_fbeta = {}'.format(metrics_result.best_fbeta))
                    s_best_roc_auc.append(metrics_result.best_roc_auc)
                    file_logger.info('best_roc_auc = {}'.format(metrics_result.best_roc_auc))
                    s_best_pr_auc.append(metrics_result.best_pr_auc)
                    file_logger.info('best_pr_auc = {}'.format(metrics_result.best_pr_auc))
                    s_best_cks.append(metrics_result.best_cks)
                    file_logger.info('best_cks = {}'.format(metrics_result.best_cks))
            meta_logger.info(dir)
            avg_TN = calculate_average_metric(s_TN)
            meta_logger.info('avg_TN = {}'.format(avg_TN))
            avg_FP = calculate_average_metric(s_FP)
            meta_logger.info('avg_FP = {}'.format(avg_FP))
            avg_FN = calculate_average_metric(s_FN)
            meta_logger.info('avg_FN = {}'.format(avg_FN))
            avg_TP = calculate_average_metric(s_TP)
            meta_logger.info('avg_TP = {}'.format(avg_TP))
            avg_precision = calculate_average_metric(s_precision)
            meta_logger.info('avg_precision = {}'.format(avg_precision))
            avg_recall = calculate_average_metric(s_recall)
            meta_logger.info('avg_recall = {}'.format(avg_recall))
            avg_fbeta = calculate_average_metric(s_fbeta)
            meta_logger.info('avg_fbeta = {}'.format(avg_fbeta))
            avg_roc_auc = calculate_average_metric(s_roc_auc)
            meta_logger.info('avg_roc_auc = {}'.format(avg_roc_auc))
            avg_pr_auc = calculate_average_metric(s_pr_auc)
            meta_logger.info('avg_pr_auc = {}'.format(avg_pr_auc))
            avg_cks = calculate_average_metric(s_cks)
            meta_logger.info('avg_cks = {}'.format(avg_cks))
            avg_best_TN = calculate_average_metric(s_best_TN)
            meta_logger.info('avg_best_TN = {}'.format(avg_best_TN))
            avg_best_FP = calculate_average_metric(s_best_FP)
            meta_logger.info('avg_best_FP = {}'.format(avg_best_FP))
            avg_best_FN = calculate_average_metric(s_best_FN)
            meta_logger.info('avg_best_FN = {}'.format(avg_best_FN))
            avg_best_TP = calculate_average_metric(s_best_TP)
            meta_logger.info('avg_best_TP = {}'.format(avg_best_TP))
            avg_best_precision = calculate_average_metric(s_best_precision)
            meta_logger.info('avg_best_precision = {}'.format(avg_best_precision))
            avg_best_recall = calculate_average_metric(s_best_recall)
            meta_logger.info('avg_best_recall = {}'.format(avg_best_recall))
            avg_best_fbeta = calculate_average_metric(s_best_fbeta)
            meta_logger.info('avg_best_fbeta = {}'.format(avg_best_fbeta))
            avg_best_roc_auc = calculate_average_metric(s_best_roc_auc)
            meta_logger.info('avg_best_roc_auc = {}'.format(avg_best_roc_auc))
            avg_best_pr_auc = calculate_average_metric(s_best_pr_auc)
            meta_logger.info('avg_best_pr_auc = {}'.format(avg_best_pr_auc))
            avg_best_cks = calculate_average_metric(s_best_cks)
            meta_logger.info('avg_best_cks = {}'.format(avg_best_cks))
            file_logger.info('Finish')
            # logger.shutdown()
            meta_logger.info('Finish')
            # meta_logger.shutdown()
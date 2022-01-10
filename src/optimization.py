import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
import warnings
import sys
import time

from constants import NUM_LABELS

class FirstLayer(nn.Module):
    def __init__(self, n_residues, n_aminos):
        super().__init__()
        self.W = nn.Parameter(torch.randn((n_residues, n_aminos), requires_grad=True))

    def forward(self, x):
        return self.W

class GumbelOnehot(nn.Module):
    def __init__(self, tau=1):
        super().__init__()
        self.tau = tau

    def update_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        x = F.gumbel_softmax(x, tau=self.tau, hard=True)
        return x.view(-1)

def make_model(n_residues, n_aminos, tau=1):
    model = nn.Sequential(
        FirstLayer(n_residues, n_aminos),
        GumbelOnehot(tau),
    )
    return model

def compute_loss(onehot, energy_matrix, device='cuda:0'):
    energy_matrix_on_device = energy_matrix.to(device)
    return onehot.T @ energy_matrix_on_device @ onehot

def soft_max_opt(energy_matrix, n_residues, n_aminos, compute_loss_func=None, loss_sign=1, lr=0.0001,
    n_epochs = 10000, verbose=True, n_runs=1, device="cuda:0" if torch.cuda.is_available() else "cpu:0", tau=1):
    model = make_model(n_residues=n_residues, n_aminos=n_aminos, tau=tau)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.8, 0.9))

    if compute_loss_func is None:
        compute_loss_func=compute_loss

    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3,
        verbose=False, threshold=0.001, threshold_mode='abs',
        cooldown=0, min_lr=0, eps=1e-08)

    scheduler2 = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda x: 1 if x % 20 == 0 else opt.param_groups[0]['lr'])

    best_loss = float("Inf")
    best_onehot = None
    for epoch in range(n_epochs):
        if tau is None:
            tau = max(0.5, np.exp(- 1e-3 * epoch))
            model[1].update_tau(tau=tau)
        onehot = model(1)
        loss = loss_sign * compute_loss_func(onehot, energy_matrix, device)
        if loss < best_loss:
            best_loss = loss
            best_onehot = onehot
        model.zero_grad()
        loss.backward()

        opt.step()
        scheduler.step(loss)
        scheduler2.step()


        if verbose:
            print(loss)

    if verbose:
        print("Best loss: {} \nfor onehot: {} ".format(best_loss, best_onehot))
    if best_onehot is None:
        raise Exception("Gumbel-Softmax optimization failed to run even one step.")

    return best_loss, best_onehot

def solve_gumbel_soft_max_opt(energy_matrix, compute_loss_func=None, minimize=True, n_aminos=NUM_LABELS,
                    n_epochs=2000, n_runs=30, device="cuda:0" if torch.cuda.is_available() else "cpu:0",
                    stat=False, verbose=False,
                    tau=1):
    try:

        loss_sign = 1 if minimize else -1

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            energy_matrix = torch.tensor(energy_matrix, dtype=torch.float).to(device)

        n_residues = energy_matrix.shape[0] // n_aminos

        start = time.time()
        results = []
        losses = []
        best_loss = float("Inf")
        for run in range(n_runs):

            loss, onehot = soft_max_opt(energy_matrix, n_residues=n_residues, n_aminos=n_aminos,
                compute_loss_func=compute_loss_func, loss_sign=loss_sign, lr=1,
                n_epochs=n_epochs, verbose=verbose, device=device,
                tau=tau
            )

            if verbose:
                print(f"Run number {run}. loss: {loss}")
            if stat:
                losses.append(loss.item())
            if loss < best_loss:
                best_loss = loss
                best_onehot = onehot
        end = time.time()

        if stat:
            losses = np.array(losses)
            std = np.std(losses)
            mean = np.mean(losses)
            print(f"Solution: {mean} +- {std}")
            return best_onehot.cpu().detach(), loss_sign * best_loss.cpu().detach(), mean, std
        else:
            return best_onehot.cpu().detach(), loss_sign * best_loss.cpu().detach()
    except:
        print(traceback.print_exc(), file=sys.stderr, flush=True)
        exit(1)

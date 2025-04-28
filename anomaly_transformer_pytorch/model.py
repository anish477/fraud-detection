import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class AnomalyAttention(nn.Module):
    def __init__(self, N, d_model):
        super(AnomalyAttention, self).__init__()
        self.d_model = d_model
        self.N = N

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Ws = nn.Linear(d_model, 1, bias=False)

        # Initialize placeholders - these will be overwritten in forward
        self.Q = torch.zeros((N, d_model))
        self.K = torch.zeros((N, d_model))
        self.V = torch.zeros((N, d_model))
        self.sigma = torch.zeros((N, 1)) # Sigma should be (N, 1) per item

        self.P = torch.zeros((N, N))
        self.S = torch.zeros((N, N))

    def forward(self, x):
        # x shape: (batch_size, N, d_model) or (N, d_model)
        self.initialize(x)
        self.P = self.prior_association()
        self.S = self.series_association()
        Z = self.reconstruction()

        return Z

    def initialize(self, x):
        # x shape: (..., N, d_model)
        self.Q = self.Wq(x)
        self.K = self.Wk(x)
        self.V = self.Wv(x)
        # Sigma should have shape (..., N, 1)
        self.sigma = torch.sigmoid(self.Ws(x)) # Apply sigmoid or softplus to ensure positivity and reasonable range

    @staticmethod
    def gaussian_kernel(mean, sigma):
        # mean: (..., N, N), sigma: (..., N, 1)
        # Ensure sigma is positive and non-zero
        sigma = torch.clamp(sigma, min=1e-6)
        return torch.exp(- mean.pow(2) / (2 * sigma.pow(2)))

    def prior_association(self):
        # sigma shape: (..., N, 1)
        # build N×N distance matrix on same device
        idx = torch.arange(self.N, device=self.sigma.device)
        p = torch.abs(idx.unsqueeze(1) - idx.unsqueeze(0)).float()    # (N, N)
        # Add batch dim if necessary
        if self.sigma.dim() > 2: # Check if batch dimension exists (e.g., shape is (batch, N, 1))
            p = p.unsqueeze(0).expand(self.sigma.shape[0], -1, -1) # Expand to (batch, N, N)
        # else: p remains (N, N) if sigma is (N, 1)

        gauss = self.gaussian_kernel(p, self.sigma)                   # (..., N, N)
        # Normalize along the last dimension (columns)
        norm_gauss = gauss / gauss.sum(dim=-1, keepdim=True)
        # Handle potential NaN resulting from zero sum (e.g., if sigma is huge)
        norm_gauss = torch.nan_to_num(norm_gauss, nan=1.0/self.N) # Replace NaN with uniform distribution
        return norm_gauss


    def series_association(self):
        # support batched inputs: Q, K have shape (..., N, d_model)
        # compute Q·Kᵀ along the last two dims and softmax over sequence dim
        K_t = self.K.transpose(-2, -1)                     # (..., d_model, N)
        sim = torch.matmul(self.Q, K_t) / math.sqrt(self.d_model)
        return F.softmax(sim, dim=-1)


    def reconstruction(self):
        # S shape: (..., N, N), V shape: (..., N, d_model)
        return torch.matmul(self.S, self.V) # Result shape: (..., N, d_model)


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, N, d_model):
        super().__init__()
        self.N, self.d_model = N, d_model

        self.attention = AnomalyAttention(self.N, self.d_model)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU())
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        x_identity = x
        attn_output = self.attention(x)
        # Residual connection and LayerNorm 1
        z = self.ln1(attn_output + x_identity)

        z_identity = z
        ff_output = self.ff(z)
        # Residual connection and LayerNorm 2
        z = self.ln2(ff_output + z_identity)

        return z


class AnomalyTransformer(nn.Module):
    def __init__(self, N, d_model, layers, lambda_):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.lambda_ = lambda_

        self.blocks = nn.ModuleList(
            [AnomalyTransformerBlock(self.N, self.d_model) for _ in range(layers)]
        )
        self.output = None

        # Store associations per forward pass
        self.P_layers = []
        self.S_layers = []

    def forward(self, x):
        # Clear previous layers' associations
        self.P_layers = []
        self.S_layers = []
        # x shape: (batch_size, N, d_model) or (N, d_model)
        current_x = x
        for block in self.blocks:
            current_x = block(current_x)
            # P and S will have shape (batch_size, N, N) or (N, N)
            self.P_layers.append(block.attention.P)
            self.S_layers.append(block.attention.S)

        self.output = current_x # Shape: (batch_size, N, d_model) or (N, d_model)
        return self.output

    def layer_association_discrepancy(self, Pl, Sl):
        # Calculate symmetrized KL divergence between P and S row-wise
        # Pl, Sl shape: (..., N, N)
        # Ensure probabilities are valid for KL divergence (sum to 1, non-negative)
        # Add small epsilon for numerical stability
        eps = 1e-7
        Pl = torch.clamp(Pl, min=eps)
        Sl = torch.clamp(Sl, min=eps)
        # Normalize again in case clamp broke it slightly (optional, depends on clamp value)
        # Pl = Pl / Pl.sum(dim=-1, keepdim=True)
        # Sl = Sl / Sl.sum(dim=-1, keepdim=True)

        # KL(P || S) for each row i. Use reduction='sum' or 'batchmean' over the distribution dim (-1)
        # F.kl_div expects input=log-prob, target=prob. D_KL(P || S) = sum P * (log P - log S)
        kl_ps = F.kl_div(Sl.log(), Pl, log_target=False, reduction='none').sum(dim=-1) # Shape: (..., N)
        # KL(S || P) for each row i. D_KL(S || P) = sum S * (log S - log P)
        kl_sp = F.kl_div(Pl.log(), Sl, log_target=False, reduction='none').sum(dim=-1) # Shape: (..., N)

        # Symmetrized KL (related to Jensen-Shannon divergence)
        layer_ad_vector = 0.5 * (kl_ps + kl_sp) # Shape: (..., N)
        # Handle potential NaNs if distributions were identical (log(1)=0) -> KL=0
        layer_ad_vector = torch.nan_to_num(layer_ad_vector, nan=0.0)
        return layer_ad_vector

    def association_discrepancy(self, P_list, S_list):
        # P_list, S_list contain tensors of shape (..., N, N)
        layer_discrepancies = []
        for P, S in zip(P_list, S_list):
            layer_discrepancies.append(self.layer_association_discrepancy(P, S)) # Each is (..., N)

        # Stack along a new dimension (layers) and then mean across layers
        # Stacked shape: (num_layers, ..., N)
        # Mean shape: (..., N)
        avg_ad_vector = torch.stack(layer_discrepancies, dim=0).mean(dim=0)
        return avg_ad_vector

    def loss_function(self, x_hat, P_list, S_list, lambda_, x):
        # x_hat, x shape: (batch_size, N, d_model)
        # P_list, S_list contain tensors of shape (batch_size, N, N)

        # Reconstruction error per item in batch (Frobenius norm over N, d_model dims)
        recon_error_per_item = torch.linalg.matrix_norm(x_hat - x, ord="fro", dim=(-2, -1)) # Shape: (batch_size,)

        # Association discrepancy (averaged over layers) per sequence position
        assoc_disc_per_pos = self.association_discrepancy(P_list, S_list) # Shape: (batch_size, N)

        # Norm of association discrepancy per item (L1 norm over sequence dimension N)
        norm_assoc_disc_per_item = torch.linalg.norm(assoc_disc_per_pos, ord=1, dim=-1) # Shape: (batch_size,)

        # Combine per item
        loss_per_item = recon_error_per_item - lambda_ * norm_assoc_disc_per_item # Shape: (batch_size,)

        # Return the mean loss across the batch
        return loss_per_item.mean() # Scalar

    def min_loss(self, x):
        # Detach Series association for minimization step
        P_list = self.P_layers
        S_list = [S.detach() for S in self.S_layers]
        # Use negative lambda_ as per original formulation for min step
        lambda_ = -self.lambda_
        return self.loss_function(self.output, P_list, S_list, lambda_, x)

    def max_loss(self, x):
        # Detach Prior association for maximization step
        P_list = [P.detach() for P in self.P_layers]
        S_list = self.S_layers
        lambda_ = self.lambda_
        # The loss function calculates Recon - lambda * AD. To maximize AD, we minimize -(Recon - lambda * AD)
        # However, the original paper maximizes lambda * AD directly. Let's stick to the paper's intention:
        # Maximize the Association Discrepancy term.
        # We need the AD term separately.
        assoc_disc_per_pos = self.association_discrepancy(P_list, S_list) # Shape: (batch_size, N)
        norm_assoc_disc_per_item = torch.linalg.norm(assoc_disc_per_pos, ord=1, dim=-1) # Shape: (batch_size,)
        # Return the mean of the discrepancy term across the batch to be maximized (by minimizing its negative)
        return norm_assoc_disc_per_item.mean() # Scalar

    def anomaly_score(self, x):
        # Assumes self.output, self.P_layers, self.S_layers are populated from a forward pass on x.
        # x shape: (N, d_model) - assuming evaluation is done one sequence at a time
        # self.output shape: (N, d_model)
        # self.P_layers, self.S_layers contain tensors of shape (N, N)

        # Calculate association discrepancy (average over layers) per sequence position
        ad = self.association_discrepancy(self.P_layers, self.S_layers) # Shape: (N,)

        # Softmax normalization of negative discrepancy (higher discrepancy -> higher score component)
        # Add small epsilon to prevent division by zero if ad is uniform
        ad_softmax = F.softmax(-ad, dim=0) # Shape: (N,)

        # Reconstruction error per sequence position (L2 norm over feature dimension d_model)
        recon_error = self.output - x # Shape: (N, d_model)
        norm_per_pos = torch.linalg.norm(recon_error, ord=2, dim=-1) # Shape: (N,)

        # Element-wise multiplication
        score = torch.mul(ad_softmax, norm_per_pos) # Shape: (N,)

        return score

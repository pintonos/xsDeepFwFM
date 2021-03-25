import numpy as np
import torch
import torch.nn.functional as F
import os

from model.layers import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron, FieldWeightedFactorizationMachine


class MultiLayerPerceptronModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)

        x = self.mlp(embed_x.view(-1, self.embed_output_dim))

        return torch.sigmoid(x.squeeze(1))


class DeepFieldWeightedFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFwFM.

    Reference:
        Deng et al., DeepLight: Deep Lightweight Feature Interactions for Accelerating CTR Predictions in Ad Serving, 2021.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, use_lw=False, use_fwlw=False, use_emb_bag=False, use_qr_emb=False, qr_collisions=4, quantize_dnn=False, batch_norm=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.use_lw = use_lw
        self.use_fwlw = use_fwlw
        self.use_emb_bag = use_emb_bag
        self.use_qr_emb = use_qr_emb
        self.mlp_dims = mlp_dims
        self.linear = FeaturesLinear(field_dims)
        self.fwfm_linear = torch.nn.Linear(embed_dim, self.num_fields, bias=False)
        self.fwfm = FieldWeightedFactorizationMachine(field_dims, embed_dim, use_emb_bag=use_emb_bag, use_qr_emb=use_qr_emb, qr_collisions=qr_collisions)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, quantize=quantize_dnn, batch_norm=batch_norm)
        self.bias = torch.nn.Parameter(torch.Tensor([0.01]))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if self.use_emb_bag or self.use_qr_emb:
            embed_x_2nd = [emb(torch.unsqueeze(x[:, i], 1).contiguous()) for i, emb in enumerate(self.fwfm.embeddings)]
        else:
            embed_x_2nd = [emb(x[:, i].contiguous()) for i, emb in enumerate(self.fwfm.embeddings)]

        fwfm_second_order = torch.sum(self.fwfm(torch.stack(embed_x_2nd)), dim=1, keepdim=True)

        if self.use_lw and not self.use_fwlw:
            x = self.linear(x) + fwfm_second_order + self.mlp(torch.cat(embed_x_2nd, 1))
        elif self.use_fwlw:
            fwfm_linear = torch.einsum('ijk,ik->ijk', [torch.stack(embed_x_2nd), self.fwfm_linear.weight])
            fwfm_first_order = torch.sum(torch.einsum('ijk->ji', [fwfm_linear]), dim=1, keepdim=True)
            x = fwfm_first_order + fwfm_second_order + self.mlp(torch.cat(embed_x_2nd, 1)) + self.bias
        else:
            x = fwfm_second_order + self.mlp(torch.cat(embed_x_2nd, 1)) + self.bias

        return torch.sigmoid(x.squeeze(1))


class FieldWeightedFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Field Weighted Factorization Machines.

    Reference:
        Pan et al., Field-weighted factorization machines for click-through rate prediction in display advertising, 2018.
    """

    def __init__(self, field_dims, embed_dim, use_lw=False, use_fwlw=False, use_emb_bag=False, use_qr_emb=False):
        super().__init__()
        self.num_fields = len(field_dims)
        self.use_lw = use_lw
        self.use_fwlw = use_fwlw
        self.use_emb_bag = use_emb_bag
        self.use_qr_emb = use_qr_emb
        self.linear = FeaturesLinear(field_dims)
        self.fwfm_linear = torch.nn.Linear(embed_dim, self.num_fields, bias=False)
        self.fwfm = FieldWeightedFactorizationMachine(field_dims, embed_dim, use_emb_bag=use_emb_bag, use_qr_emb=use_qr_emb)
        self.bias = torch.nn.Parameter(torch.Tensor([0.01]))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if self.use_emb_bag or self.use_qr_emb:
            embed_x_2nd = [emb(torch.unsqueeze(x[:, i], 1)) for i, emb in enumerate(self.fwfm.embeddings)]
        else:
            embed_x_2nd = [emb(x[:, i]) for i, emb in enumerate(self.fwfm.embeddings)]

        fwfm_second_order = torch.sum(self.fwfm(torch.stack(embed_x_2nd)), dim=1, keepdim=True)

        if self.use_lw and not self.use_fwlw:
            x = self.linear(x) + fwfm_second_order + self.bias
        elif self.use_fwlw:
            fwfm_linear = torch.einsum('ijk,ik->ijk', [torch.stack(embed_x_2nd), self.fwfm_linear.weight])
            fwfm_first_order = torch.sum(torch.einsum('ijk->ji', [fwfm_linear]), dim=1, keepdim=True)
            x = fwfm_first_order + fwfm_second_order + self.bias
        else:
            x = fwfm_second_order + self.bias

        return torch.sigmoid(x.squeeze(1))


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy, epoch, optimizer, loss):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            if not os.path.exists(os.path.dirname(self.save_path)):
                os.makedirs(os.path.dirname(self.save_path))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False




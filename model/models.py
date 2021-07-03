import numpy as np
import torch
import torch.nn.functional as F
import os
from timeit import default_timer as timer

from model.layers import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron, FieldWeightedFactorizationMachine, FactorizationMachine, CompressedInteractionNetwork


'''
Reference: https://github.com/rixwew/pytorch-fm
'''


class MultiLayerPerceptronModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout=0.0, use_qr_emb=False, qr_collisions=4, batch_norm=True, return_raw_logits=False):
        super().__init__()
        self.use_qr_emb = use_qr_emb
        self.qr_collisions = qr_collisions
        self.embeddings = FeaturesEmbedding(field_dims, embed_dim, use_qr_emb, qr_collisions)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, batch_norm=batch_norm)
        self.mlp_dims = mlp_dims
        self.return_raw_logits = return_raw_logits

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embeddings(x)

        x = self.mlp(embed_x.view(-1, self.embed_output_dim))

        if self.return_raw_logits:
            return x.squeeze(1)
        return torch.sigmoid(x.squeeze(1))


class DeepFieldWeightedFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFwFM.

    Reference:
        Deng et al., DeepLight: Deep Lightweight Feature Interactions for Accelerating CTR Predictions in Ad Serving, 2021.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout=0.0, use_lw=False, use_fwlw=False, use_qr_emb=False, qr_collisions=4, quantize_dnn=False, batch_norm=True, return_raw_logits=False):
        super().__init__()
        self.num_fields = len(field_dims)
        self.use_lw = use_lw
        self.use_fwlw = use_fwlw
        self.mlp_dims = mlp_dims
        self.use_qr_emb = use_qr_emb
        self.qr_collisions = qr_collisions
        self.linear = FeaturesLinear(field_dims)
        self.embeddings = FeaturesEmbedding(field_dims, embed_dim, use_qr_emb, qr_collisions)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.fwfm_linear = torch.nn.Linear(embed_dim, self.num_fields, bias=False)
        self.fwfm = FieldWeightedFactorizationMachine(field_dims)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, quantize=quantize_dnn, batch_norm=batch_norm)
        self.bias = torch.nn.Parameter(torch.Tensor([0.01]))
        self.return_raw_logits = return_raw_logits

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embeddings(x)

        fwfm_second_order = torch.sum(self.fwfm(torch.transpose(embed_x, 0, 1)), dim=1, keepdim=True)

        if self.use_lw and not self.use_fwlw:
            x = self.linear(x) + fwfm_second_order + self.mlp(embed_x.view(-1, self.embed_output_dim))
        elif self.use_fwlw:
            fwfm_linear = torch.einsum('ijk,ik->ijk', [torch.transpose(embed_x, 0, 1), self.fwfm_linear.weight])
            fwfm_first_order = torch.sum(torch.einsum('ijk->ji', [fwfm_linear]), dim=1, keepdim=True)
            x = fwfm_first_order + fwfm_second_order + self.mlp(embed_x.view(-1, self.embed_output_dim)) + self.bias
        else:
            x = fwfm_second_order + self.mlp(embed_x.view(-1, self.embed_output_dim)) + self.bias

        if self.return_raw_logits:
            return x.squeeze(1)
        return torch.sigmoid(x.squeeze(1))


class FieldWeightedFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Field Weighted Factorization Machines.

    Reference:
        Pan et al., Field-weighted factorization machines for click-through rate prediction in display advertising, 2018.
    """

    def __init__(self, field_dims, embed_dim, use_lw=False, use_fwlw=False, use_qr_emb=False, qr_collisions=4):
        super().__init__()
        self.num_fields = len(field_dims)
        self.use_lw = use_lw
        self.use_fwlw = use_fwlw
        self.use_qr_emb = use_qr_emb
        self.qr_collisions = qr_collisions
        self.linear = FeaturesLinear(field_dims)
        self.embeddings = FeaturesEmbedding(field_dims, embed_dim, use_qr_emb, qr_collisions)
        self.fwfm_linear = torch.nn.Linear(embed_dim, self.num_fields, bias=False)
        self.fwfm = FieldWeightedFactorizationMachine(field_dims)
        self.bias = torch.nn.Parameter(torch.Tensor([0.01]))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = torch.transpose(self.embeddings(x), 0, 1)

        start = timer()
        fwfm_second_order = torch.sum(self.fwfm(embed_x), dim=1, keepdim=True)
        end = timer()
        print(end - start)

        if self.use_lw and not self.use_fwlw:
            x = self.linear(x) + fwfm_second_order + self.bias
        elif self.use_fwlw:
            fwfm_linear = torch.einsum('ijk,ik->ijk', [embed_x, self.fwfm_linear.weight])
            fwfm_first_order = torch.sum(torch.einsum('ijk->ji', [fwfm_linear]), dim=1, keepdim=True)
            x = fwfm_first_order + fwfm_second_order + self.bias
        else:
            x = fwfm_second_order + self.bias

        return torch.sigmoid(x.squeeze(1))


class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.
    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout=0.0):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.
    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim, use_lw=False, use_qr_emb=False, qr_collisions=4):
        super().__init__()
        self.use_lw = use_lw
        self.use_qr_emb = use_qr_emb
        self.qr_collisions = qr_collisions
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, use_qr_emb=use_qr_emb, qr_collisions=qr_collisions)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if self.use_lw:
            x = self.linear(x) + self.fm(self.embedding(x))
        else:
            x = self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


class LogisticRegressionModel(torch.nn.Module):
    """
    A pytorch implementation of Logistic Regression.
    """

    def __init__(self, field_dims):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sigmoid(self.linear(x).squeeze(1))


class ExtremeDeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.
    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class EarlyStopper(object):

    def __init__(self, num_trials, save_path, accuracy=0):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = accuracy
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
                'loss': loss,
                'accuracy': accuracy
            }, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False




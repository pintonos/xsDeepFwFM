import numpy as np
import torch
import torch.nn.functional as F

from model.layers import FeaturesLinear, FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron, FieldWeightedFactorizationMachine


class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.

    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, use_linear=False):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.use_linear = use_linear
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        if self.use_linear:
            x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        else:
            x = self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return torch.sigmoid(x.squeeze(1))


class DeepFieldWeightedFactorizationMachineModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, use_lw=False, use_fwlw=False, use_emb_bag=False, use_qr_emb=False, quantize_dnn=False, batch_norm=False):
        super().__init__()
        self.num_fields = len(field_dims)
        self.use_lw = use_lw
        self.use_fwlw = use_fwlw
        self.use_emb_bag = use_emb_bag
        self.use_qr_emb = use_qr_emb
        self.linear = FeaturesLinear(field_dims)
        self.fwfm_linear = torch.nn.Linear(embed_dim, self.num_fields, bias=False)
        self.fwfm = FieldWeightedFactorizationMachine(field_dims, embed_dim, use_emb_bag=use_emb_bag, use_qr_emb=use_qr_emb)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, quantize=quantize_dnn, batch_norm=batch_norm)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if self.use_emb_bag or self.use_qr_emb:
            embed_x = [self.fwfm.embeddings[i](torch.unsqueeze(x[:, i], 1)) for i in range(self.num_fields)]
        else:
            embed_x = [self.fwfm.embeddings[i](x[:, i]) for i in range(self.num_fields)]

        fwfm_second_order = torch.sum(self.fwfm(torch.stack(embed_x)), dim=1, keepdim=True)

        if self.use_lw:
            x = self.linear(x) + fwfm_second_order + self.mlp(torch.cat(embed_x, 1))
        elif self.use_fwlw:
            fwfm_linear = torch.einsum('ijk,ik->ijk', [torch.stack(embed_x), self.fwfm_linear.weight])
            fwfm_first_order = torch.sum(torch.einsum('ijk->ji', [fwfm_linear]), dim=1, keepdim=True)
            x = fwfm_first_order + fwfm_second_order + self.mlp(torch.cat(embed_x, 1))
        else:
            x = fwfm_second_order + self.mlp(torch.cat(embed_x, 1))

        return torch.sigmoid(x.squeeze(1))


class FieldWeightedFactorizationMachineModel(torch.nn.Module):

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

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if self.use_emb_bag or self.use_qr_emb:
            embed_x = [self.fwfm.embeddings[i](torch.unsqueeze(x[:, i], 1)) for i in range(self.num_fields)]
        else:
            embed_x = [self.fwfm.embeddings[i](x[:, i]) for i in range(self.num_fields)] # TODO most computation here?

        fwfm_second_order = torch.sum(self.fwfm(torch.stack(embed_x)), dim=1, keepdim=True)

        if self.use_lw:
            x = self.linear(x) + fwfm_second_order
        elif self.use_fwlw:
            fwfm_linear = torch.einsum('ijk,ik->ijk', [torch.stack(embed_x), self.fwfm_linear.weight])
            fwfm_first_order = torch.sum(torch.einsum('ijk->ji', [fwfm_linear]), dim=1, keepdim=True)
            x = fwfm_first_order + fwfm_second_order
        else:
            x = fwfm_second_order

        return torch.sigmoid(x.squeeze(1))


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim, use_lw=False):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.use_lw = use_lw
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




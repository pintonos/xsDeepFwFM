import numpy as np
import torch
import torch.nn.functional as F
from model.QREmbedding import QREmbedding

'''
Reference: https://github.com/rixwew/pytorch-fm
'''


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, use_qr_emb=False, qr_collisions=4):
        super().__init__()

        self.use_qr_emb = use_qr_emb

        if use_qr_emb:
            self.embeddings = self.create_emb(embed_dim, field_dims, use_qr_emb, qr_collisions=qr_collisions)
        else:
            self.embeddings = torch.nn.ModuleList([
                torch.nn.Embedding(field_dim, embed_dim) for field_dim in field_dims
            ])
            for embedding in self.embeddings:
                torch.nn.init.xavier_uniform_(embedding.weight.data)

    def create_emb(self, m, ln, qr=False, qr_operation="mult", qr_collisions=4, qr_threshold=200):
        emb_l = torch.nn.ModuleList()
        for i in range(0, ln.size):
            n = ln[i]
            # construct embedding operator
            if qr and n > qr_threshold:
                EE = QREmbedding(n, m, qr_collisions, operation=qr_operation)
            else:
                EE = torch.nn.Embedding(n, m)
                torch.nn.init.xavier_uniform_(EE.weight.data)
            emb_l.append(EE)
        return emb_l

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = [emb(x[:, i].contiguous()) for i, emb in enumerate(self.embeddings)]

        return torch.stack(embed_x, axis=1)


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0.0, output_layer=True, quantize=False, batch_norm=True):
        super().__init__()
        layers = list()
        if not dropout == 0.0:
            layers.append(torch.nn.Dropout(p=dropout))
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if not dropout == 0.0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self.quantize = quantize
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        if self.quantize:
            x = self.quant(x)
            return self.dequant(self.mlp(x))
        else:
            return self.mlp(x)


class FieldWeightedFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims):
        super().__init__()
        self.num_fields = len(field_dims)
        self.field_cov = torch.nn.Linear(self.num_fields, self.num_fields, bias=False)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(num_fields, batch_size, embed_dim)``
        """
        outer_fm = torch.einsum('kij,lij->klij', x, x)
        outer_fwfm = torch.einsum('klij,kl->klij', outer_fm, (self.field_cov.weight.t() + self.field_cov.weight) * 0.5)
        fwfm_second_order = (torch.sum(torch.sum(outer_fwfm, 0), 0) - torch.sum(torch.einsum('kkij->kij', outer_fwfm), 0)) * 0.5

        return fwfm_second_order
        

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))
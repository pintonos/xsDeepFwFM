import numpy as np
import torch
import torch.nn.functional as F
from model.QREmbeddingBag import QREmbeddingBag


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

    def __init__(self, field_dims, embed_dim, use_emb_bag=False, use_qr_emb=False, qr_collisions=4):
        super().__init__()
        
        self.use_emb_bag = use_emb_bag
        self.use_qr_emb = use_qr_emb

        if use_emb_bag or use_qr_emb:
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
                EE = QREmbeddingBag(n, m, qr_collisions,
                                    operation=qr_operation, mode="sum", sparse=False)
            else:
                EE = torch.nn.EmbeddingBag(n, m, mode="sum", sparse=False)
                torch.nn.init.xavier_uniform_(EE.weight.data)
            emb_l.append(EE)
        return emb_l

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if self.use_emb_bag or self.use_qr_emb:
            embed_x = [emb(torch.unsqueeze(x[:, i], 1).contiguous()) for i, emb in enumerate(self.embeddings)]
        else:
            embed_x = [emb(x[:, i].contiguous()) for i, emb in enumerate(self.embeddings)]
        return embed_x


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True, quantize=False, batch_norm=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
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
        outer_fm = torch.einsum('kij,lij->klij', x, x)
        outer_fwfm = torch.einsum('klij,kl->klij', outer_fm, (self.field_cov.weight.t() + self.field_cov.weight) * 0.5)
        fwfm_second_order = (torch.sum(torch.sum(outer_fwfm, 0), 0) - torch.sum(torch.einsum('kkij->kij', outer_fwfm), 0)) * 0.5

        return fwfm_second_order
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Quotient-Remainder Trick
#
# Description: Applies quotient remainder-trick to embeddings to reduce
# embedding sizes.
#
# References:
# [1] Hao-Jun Michael Shi, Dheevatsa Mudigere, Maxim Naumov, Jiyan Yang,
# "Compositional Embeddings Using Complementary Partitions for Memory-Efficient
# Recommendation Systems", CoRR, arXiv:1909.02107, 2019


from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class QREmbedding(nn.Module):
    r"""
    Args:
        num_categories (int): total number of unique categories. The input indices must be in
                              0, 1, ..., num_categories - 1.
        embedding_dim (list): list of sizes for each embedding vector in each table. If ``"add"``
                              or ``"mult"`` operation are used, these embedding dimensions must be
                              the same. If a single embedding_dim is used, then it will use this
                              embedding_dim for both embedding tables.
        num_collisions (int): number of collisions to enforce.
        operation (string, optional): ``"concat"``, ``"add"``, or ``"mult". Specifies the operation
                                      to compose embeddings. ``"concat"`` concatenates the embeddings,
                                      ``"add"`` sums the embeddings, and ``"mult"`` multiplies
                                      (component-wise) the embeddings.
                                      Default: ``"mult"``
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (string, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 ``"sum"`` computes the weighted sum, taking :attr:`per_sample_weights`
                                 into consideration. ``"mean"`` computes the average of the values
                                 in the bag, ``"max"`` computes the max value over each bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor. See
                                 Notes for more details regarding sparse gradients. Note: this option is not
                                 supported when ``mode="max"``.

    Attributes:
        weight (Tensor): the learnable weights of each embedding table is the module of shape
                         `(num_embeddings, embedding_dim)` initialized using a uniform distribution
                         with sqrt(1 / num_categories).

    Inputs: :attr:`input` (LongTensor), :attr:`offsets` (LongTensor, optional), and
        :attr:`per_index_weights` (Tensor, optional)

        - If :attr:`input` is 2D of shape `(B, N)`,

          it will be treated as ``B`` bags (sequences) each of fixed length ``N``, and
          this will return ``B`` values aggregated in a way depending on the :attr:`mode`.
          :attr:`offsets` is ignored and required to be ``None`` in this case.

        - If :attr:`input` is 1D of shape `(N)`,

          it will be treated as a concatenation of multiple bags (sequences).
          :attr:`offsets` is required to be a 1D tensor containing the
          starting index positions of each bag in :attr:`input`. Therefore,
          for :attr:`offsets` of shape `(B)`, :attr:`input` will be viewed as
          having ``B`` bags. Empty bags (i.e., having 0-length) will have
          returned vectors filled by zeros.

        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be ``1``. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not ``None``. Only supported for ``mode='sum'``.


    Output shape: `(B, embedding_dim)`

    """
    __constants__ = ['num_categories', 'embedding_dim', 'num_collisions',
                     'operation', 'max_norm', 'norm_type', 'scale_grad_by_freq',
                     'mode', 'sparse']

    def __init__(self, num_categories, embedding_dim, num_collisions,
                 operation='mult', max_norm=None, norm_type=2.,
                 scale_grad_by_freq=False, _weight=None):
        super(QREmbedding, self).__init__()

        assert operation in ['concat', 'mult', 'add'], 'Not valid operation!'

        self.num_categories = num_categories
        if isinstance(embedding_dim, int) or len(embedding_dim) == 1:
            self.embedding_dim = [embedding_dim, embedding_dim]
        else:
            self.embedding_dim = embedding_dim
        self.num_collisions = num_collisions
        self.operation = operation
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        if self.operation == 'add' or self.operation == 'mult':
            assert self.embedding_dim[0] == self.embedding_dim[1], \
                'Embedding dimensions do not match!'

        self.num_embeddings = [int(np.ceil(num_categories / num_collisions)),
            num_collisions]

        if _weight is None:
            self.weight_q = Parameter(torch.Tensor(self.num_embeddings[0], self.embedding_dim[0]))
            self.weight_r = Parameter(torch.Tensor(self.num_embeddings[1], self.embedding_dim[1]))
            self.reset_parameters()
        else:
            assert list(_weight[0].shape) == [self.num_embeddings[0], self.embedding_dim[0]], \
                'Shape of weight for quotient table does not match num_embeddings and embedding_dim'
            assert list(_weight[1].shape) == [self.num_embeddings[1], self.embedding_dim[1]], \
                'Shape of weight for remainder table does not match num_embeddings and embedding_dim'
            self.weight_q = Parameter(_weight[0])
            self.weight_r = Parameter(_weight[1])

    def reset_parameters(self):
        nn.init.uniform_(self.weight_q, np.sqrt(1 / self.num_categories))
        nn.init.uniform_(self.weight_r, np.sqrt(1 / self.num_categories))

    def forward(self, input, offsets=None):
        input_q = (input // self.num_collisions).long()
        input_r = torch.remainder(input, self.num_collisions).long()

        embed_q = F.embedding(input_q, self.weight_q, offsets, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq)
        embed_r = F.embedding(input_r, self.weight_r, offsets, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq)

        if self.operation == 'concat':
            embed = torch.cat((embed_q, embed_r), dim=1)
        elif self.operation == 'add':
            embed = embed_q + embed_r
        elif self.operation == 'mult':
            embed = embed_q * embed_r

        return embed

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        return s.format(**self.__dict__)

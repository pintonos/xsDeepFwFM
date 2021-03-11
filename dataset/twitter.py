import math
import shutil
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import lmdb
import numpy as np
import torch.utils.data
from tqdm import tqdm

import pandas as pd

sparse_features = ['a_is_verified', 'b_is_verified', 'b_follows_a', 'id',
                   'language', 'tweet_type', 'media', 'tweet_id', 'a_user_id', 'b_user_id', 'domains', 'links',
                   'hashtags', 'tr', 'dt_day', 'dt_dow', 'dt_hour', 'a_count_combined', 'a_user_fer_count_delta_time',
                   'a_user_fing_count_delta_time',
                   'a_user_fering_count_delta_time', 'a_user_fing_count_mode',
                   'a_user_fer_count_mode', 'a_user_fering_count_mode', 'count_ats',
                   'count_char', 'count_words', 'tw_hash', 'tw_freq_hash', 'tw_first_word',
                   'tw_second_word', 'tw_last_word', 'tw_llast_word', 'tw_hash0',
                   'tw_hash1', 'tw_rt_uhash']  # = categorical features

dense_features = ['timestamp', 'a_follower_count', 'a_following_count', 'a_account_creation',
                  'b_follower_count', 'b_following_count', 'b_account_creation',
                  'len_hashtags', 'len_domains', 'len_links', 'tw_len']  # = numerical features

label_names = ['reply', 'retweet', 'retweet_comment', 'like']


class TwitterDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path=None, cache_path='.twitter', rebuild_cache=False, min_threshold=4, twitter_label='like'):
        self.NUM_LABELS = 4
        self.NUM_FEATS = 47
        self.NUM_INT_FEATS = 11
        self.min_threshold = min_threshold # TODO wanted?
        self.LABEL_IDX = ['reply', 'retweet', 'retweet_comment', 'like'].index(twitter_label)
        df = pd.read_parquet(dataset_path)
        df.fillna(0, inplace=True)
        df = df[label_names + dense_features + sparse_features]
        df.to_csv('./tmp.txt', index=False, sep='\t', header=False)
        dataset_path = './tmp.txt'
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
        os.remove('./tmp.txt')
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)[self.NUM_LABELS:]

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        return np_array[self.NUM_LABELS:], np_array[self.LABEL_IDX]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        feat_mapper, defaults = self.__get_feat_mapper(path)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(self.NUM_LABELS + self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[i] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create twitter dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + self.NUM_LABELS:
                    continue
                for i in range(self.NUM_LABELS, self.NUM_INT_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + self.NUM_LABELS):
                    feat_cnts[i][values[i]] += 1
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create twitter dataset cache: setup lmdb')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + self.NUM_LABELS:
                    continue
                np_array = np.zeros(self.NUM_LABELS + self.NUM_FEATS, dtype=np.uint32)
                np_array[self.LABEL_IDX] = int(values[self.LABEL_IDX])
                for i in range(self.NUM_LABELS, self.NUM_INT_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + self.NUM_LABELS):
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer

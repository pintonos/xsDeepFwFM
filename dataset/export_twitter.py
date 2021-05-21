import math
import shutil
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import os

import lmdb
import numpy as np
import torch.utils.data
from tqdm import tqdm

import pandas as pd
import dask.dataframe as dd

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


df = pd.read_parquet('../../twitter_final_merged.parquet')
df = df.fillna(0)
df = df[label_names + dense_features + sparse_features]
#df.compute().to_csv('../../twitter.txt', index=False, sep='\t', header=False)
df.to_csv('twitter.txt', index=False, sep='\t', header=False)
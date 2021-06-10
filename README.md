# Extreme Small Deep Field-Weighted Factorization Machine - xsDeepFwFM
Acceleration and Compression of DeepFwFM.

Original paper (DeepFwFM):
```
@inproceedings{deeplight,
  title={DeepLight: Deep Lightweight Feature Interactions for Accelerating CTR Predictions in Ad Serving},
  author={Wei Deng and Junwei Pan and Tian Zhou and Deguang Kong and Aaron Flores and Guang Lin},
  booktitle={International Conference on Web Search and Data Mining (WSDM'21)},
  year={2021}
}
```

In this repository additional model compression and acceleration will be conducted. 

Used techniques:
- QR Embeddings
- Quantization
- Knowledge Distillation

Evaluation on the Criteo dataset and the Twitter dataset given by the RecSys 2020 Challenge.

This repository is part of my master thesis.


## Results

### Criteo Dataset
- Split: first 6 days for training & last day for 50% validation and 50% test
- Epochs: 50 with early stopping
- Batch Size (Training): 2048
- Intel Xeon E3-1231v3 & NVIDIA GTX 970

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Notes |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|--------|
| EmbeddingBag    | None          | (0,0,0)       | 0.4473    | 0.8058 | 0.6106    | 21.71 | 11,956,823    |  47.84    |
| EmbeddingBag    | None          | (400,400,400) | 0.4447    | 0.8085 | 0.6158    | 22.16 | 12,436,824    |  49.780   | 0.0 dropout, 1e-6
| QR EmbeddingBag | None          | (400,400,400) | 0.4458    | 0.8072 | 0.6140    | 21.97 |  4,294,354    |  17.217   | 4 collisions, 0.0 dropout, 1e-6
| QR EmbeddingBag | None          | (400,400,400) | 0.4517    | 0.8008 | 0.6033    | 20.94 |  1,771,504    |   7.125   | 60 collisions TODO
| EmbeddingBag    | Dynamic       | (400,400,400) | 0.4455    | 0.8076 | 0.6143    | 22.02 | 12,436,824    |  48.35    | TODO
| EmbeddingBag    | Static        | (400,400,400) | 0.4456    | 0.8076 | 0.6142    | 22.01 | 12,436,824    |  24.46    | TODO
| EmbeddingBag    | QAT           | (400,400,400) | 0.4459    | 0.8073 | 0.6135    | 21.94 | 12,436,824    |  24.46    | TODO       


#### Embeddings - Latency
| Embedding       |  # Deep Nodes | 1 (CPU) | 8     | 16     | 32     | 64     | 128    | 256     | 512     | 1024    | 512 (GPU) | 1024    | 2048    | 4096    | Notes |
|-----------------|---------------|---------|-------|--------|--------|--------|--------|---------|---------|---------|-----------|---------|---------|---------|-------|
| EmbeddingBag    | (0,0,0)       | 1.681   | 1.996 | 2.210  | 4.046  | 6.388  | 11.334 | 20.628* | 41.170* | 107.660 | 5.103     | 6.657   | 10.067  | 28.449  |
| EmbeddingBag    | (400,400,400) | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  | * is fastest
| QR EmbeddingBag | (400,400,400) | 5.710   | 8.330 | 10.860 | 17.814 | 31.444 | 61.628 | 112.410 | 217.246 | 422.336*| 9.995     | 12.156  | 15.940  | 31.046  |
| QR EmbeddingBag | (400,400,400) | 4.584   | 5.796 | 8.068  | 14.502 | 25.028 | 46.712 | 90.102  | 178.260*| 364.044 | 8.475     | 10.706  | 13.831  | 20.780  |

#### Quantization - Latency
| Embedding       | Quantization  | 1 (CPU) | 16    | 32     | 64     | 128    | 256      | 512     | 1024    | Notes |
|-----------------|---------------|---------|-------|--------|--------|--------|----------|---------|---------|-------|
| EmbeddingBag    | None          | 2.396   | 3.463 |        | 44.724 | 86.406 | 167.728  | 331.486*| 680.618 |
| EmbeddingBag    | Dynamic       | 2.376   | 3.516 | 5.858  | 9.132  | 15.876 | 27.849   | 54.623  | 118.678 | room was hot, measure again, average 5 times?
| EmbeddingBag    | Static        | 5.462   | 5.934 | 7.844  | 11.139 | 15.902 | 26.946   | 48.102  | 98.914  | same?
| EmbeddingBag    | QAT           | 6.057   | 8.724 |        | 14.047 | 18.672 | 32.102   | 61.676  | 144.016 |

#### Knowledge Distillation - Latency
| Embedding       |  # Deep Nodes | 1 (CPU) | 8     | 16     | 32     | 64     | 128    | 256     | 512     | 1024    | 512 (GPU) | 1024    | 2048    | 4096    | Notes |
|-----------------|---------------|---------|-------|--------|--------|--------|--------|---------|---------|---------|-----------|---------|---------|---------|-------|
| EmbeddingBag    | (400,400,400) | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  | * is fastest
| EmbeddingBag    | (128,128,128) | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  | 
| EmbeddingBag    | (128,128,128) | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  | 

 
#### Ensembles
| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Notes |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|--------|
| QR EmbeddingBag | Dynamic       | (400,400,400) | 0.4468    | 0.8062 | 0.6121    | 21.80 | Nan           |  15.789   | 4 coll
| QR EmbeddingBag | Static        | (400,400,400) | 0.4468    | 0.8061 | 0.6120    | 21.79 | Nan           |  15.777   | 4 coll
| QR EmbeddingBag | Dynamic       | (400,400,400) | 0.4489    | 0.8039 | 0.6085    | 21.44 | Nan           |   7.655   | 16 coll
| QR EmbeddingBag | Static        | (400,400,400) | 0.4489    | 0.8038 | 0.6083    | 21.42 | Nan           |   7.643   | 16 coll


#### Ensembles - Latency

| Embedding       | Quantization  | # Deep Nodes  | 1 (CPU) | 16    | 32     | 64     | 128    | 256      | 512     | 1024    | Notes |
|-----------------|---------------|---------------|---------|-------|--------|--------|--------|----------|---------|---------|-------|
| QR EmbeddingBag | Dynamic       | (400,400,400) | 5.180   | 6.494 | 8.038  | 11.406 | 16.702 | 37.160   | 58.712 *| 121.752 | 4 
| QR EmbeddingBag | Static        | (400,400,400) | 5.166   | 6.770 | 8.414  | 11.386 | 16.178 | 28.106   | 50.376 *| 103.644 | 4
| QR EmbeddingBag | Dynamic       | (400,400,400) | 4.178   | 5.082 | 7.232  | 10.446 | 16.482 | 30.586   | 56.124 *| 119.338 | 16
| QR EmbeddingBag | Static        | (400,400,400) | 5.516   | 6.318 | 8.128  | 11.036 | 16.990 | 29.040   | 52.546 *| 109.302 | 16


### Twitter Dataset, 30 threshold

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) | Notes      |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|------------|
| Embedding       | None          | (400,400,400) | 0.3173    | 0.9365 |  0.9026   | 53.29 | 62,390,100    |  249.596  | Like, 1 Epoch...then OF
| EmbeddingBag    | None          | (400,400,400) | 0.1018    | 0.8407 |  0.1279   | 14.38 | 62,390,100    |  249.596  | Reply, 1 Epoch
| EmbeddingBag    | None          | (400,400,400) | 0.2275    | 0.8641 |  0.4777   | 29.12 | 62,390,100    |  249.596  | Retweet, 1 Epoch
| EmbeddingBag    | None          | (400,400,400) | 0.0385    | 0.8053 |  0.0280   |  5.18 | 62,390,100    |  249.596  | Retweet with comment, 1 Epoch
| EmbeddingBag    | None          | (400,400,400) | 0.0372    | 0.8069 |  0.0293   |  8.49 | 62,390,100    |  249.596  | Retweet with comment, 6 Epochs, then OF

### K80, 7 threshold

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) | Notes      |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|------------|
| EmbeddingBag    | None          | (400,400,400) | 0.3285    | 0.9366 |  0.9061   | 51.64 | 348,767,961   | 1395.106  | Like, 2 Epochs
| EmbeddingBag    | None          | (400,400,400) | 0.0967    | 0.8494 |  0.1498   | 18.62 | 348,767,961   | 1395.106  | Reply, 3 Epochs no OF
| EmbeddingBag    | None          | (400,400,400) | 0.2176    | 0.8777 |  0.5306   | 32.14 | 348,767,961   | 1395.106  | Retweet, 2 Epochs, minimal improvement
| EmbeddingBag    | None          | (400,400,400) | 0.0360    | 0.8120 |  0.0402   | 11.45 | 348,767,961   | 1395.106  | Retweet with comment, 3 Epochs no OF

### K80, 15 threshold
- Epochs: 10 with early stopping

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) | Notes      |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|------------|
| EmbeddingBag    | None          | (400,400,400) | 0.3220    | 0.9388 |  0.9086   | 52.59 | 131,937,765   |  527.786  | Like, 2 Epoch, 0.2 dropout
| EmbeddingBag    | None          | (400,400,400) | 0.3324    | 0.9388 |  0.9083   | 51.06 | 131,937,765   |  527.786  | Like, 3 Epochs, 0.5 dropout TODO more training
| EmbeddingBag    | None          | (400,400,400) | 0.2155    | 0.8814 |  0.5378   | 32.81 | 131,937,765   |  527.786  | Retweet, early stopped: 8
| EmbeddingBag    | None          | (400,400,400) | 0.2155    | 0.8814 |  0.5378   | 32.81 | 131,937,765   |  527.786  | Retweet with comment, epoch +4 looking good 


## References
- https://github.com/rixwew/pytorch-fm
- https://github.com/WayneDW/DeepLight_Deep-Lightweight-Feature-Interactions
- https://github.com/peterliht/knowledge-distillation-pytorch
- https://github.com/rapidsai/deeplearning
- https://github.com/facebookresearch/dlrm
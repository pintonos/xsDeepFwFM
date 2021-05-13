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

In this repository additional model compression and acceleration will be conducted. Evaluation on the Criteo dataset and the Twitter dataset given by the RecSys 2020 Challenge.

This repository is part of my master thesis.


## Results

### Criteo Dataset
- Split: first 6 days for training & last day for 50% validation and 50% test
- Epochs: 5
- Batch Size (Training): 2048
- Intel Xeon E3-1231v3 & NVIDIA GTX 970

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Notes |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|--------|
| Embedding       | None          | (0,0,0)       | 0.4481    | 0.8049 | 0.6092    | 21.56 | 11,956,823    |  47.842   | 
| EmbeddingBag    | None          | (0,0,0)       | 0.4480    | 0.8050 | 0.6093    | 21.58 | 11,956,823    |  47.842   | 
| Embedding       | None          | (400,400,400) | 0.4457    | 0.8075 | 0.6141    | 21.99 | 12,436,824    |  49.780   | 
| EmbeddingBag    | None          | (400,400,400) | 0.4455    | 0.8077 | 0.6143    | 22.02 | 12,436,824    |  49.780   | 
| QR EmbeddingBag | None          | (400,400,400) | 0.4468    | 0.8062 | 0.6121    | 21.80 |  4,294,354    |  17.216   | 4 collisions 
| QR EmbeddingBag | None          | (400,400,400) | 0.4488    | 0.8039 | 0.6085    | 21.44 |  2,260,814    |   9.082   | 16 collisions            
| EmbeddingBag    | Dynamic       | (400,400,400) | 0.4455    | 0.8076 | 0.6143    | 22.02 | 11,959,223    |  48.35    |  
| EmbeddingBag    | Static        | (400,400,400) | 0.4456    | 0.8076 | 0.6142    | 22.01 | NaN           |  24.46    | 
| EmbeddingBag    | QAT           | (400,400,400) | 0.4459    | 0.8073 | 0.6135    | 21.94 | NaN           |  24.46    | 5 Epochs
| EmbeddingBag    | None          | (128,128,128) | 0.4502    | 0.8025 | 0.6060    | 21.20 | 12,040,792    |  48.189   | KD  
| EmbeddingBag    | None          | (128,128,128) | 0.4460    | 0.8071 | 0.6134    | 21.93 | 12,040,792    |  48.189   | without KD 
| EmbeddingBag    | None          | (64,64,64)    | 0.4502    | 0.8024 | 0.6058    | 21.20 | 11,990,616    |  47.987   | KD, similar to 128?
| EmbeddingBag    | None          | (64,64,64)    | 0.4463    | 0.8068 | 0.6125    | 21.88 | 11,990,616    |  47.987   | without KD 
| EmbeddingBag    | None          | (32,32,32)    | 0.4504    | 0.8022 | 0.6055    | 21.16 | 11,971,672    |  47.911   | KD
| EmbeddingBag    | None          | (32,32,32)    | 0.4471    | 0.8060 | 0.6111    | 21.75 | 11,971,672    |  47.911   | without KD 
| EmbeddingBag    | None          | (16,16)       | 0.4507    | 0.8019 | 0.6048    | 21.11 | 11,963,432    |  47.875   | KD 
| EmbeddingBag    | None          | (16,16)       | 0.4474    | 0.8057 | 0.6107    | 21.70 | 11,963,432    |  47.875   | without KD 
| EmbeddingBag    | None          | (64,64,64)    | 0.5319    | 0.8032 | 0.6066    |  6.90 | 11,990,616    |  47.987   | KD, new technique
| EmbeddingBag    | None          | (64,64,64)    | 0.5197    | 0.8056 | 0.6106    |  9.04 | 11,990,616    |  47.987   | KD, new technique 5 ep

#### Knowledge Distillation - TEST

- DNN only
- 1 Epoch
- TODO 2 Epochs

| # Deep Nodes  | LogLoss   | AUC    |  Notes |
|---------------|-----------|--------|--------|
| (128,128,128) | 0.4506    | 0.8021 | without KD 
| (128,128,128) | 0.4489    | 0.8042 | without KD, 2 epochs
| (128,128,128) | 0.4540    | 0.8016 | KD a = 0.5 
| (128,128,128) | 0.4499    | 0.8039 | KD a = 0.3, 2 epochs, new
| (128,128,128) | 0.4479    | 0.8056 | KD a = 0.3, 5 epochs, old
| (128,128,128) | 0.4520    | 0.8008 | KD a = 0.9, 2 epochs, old

#### Embeddings
| Embedding       |  # Deep Nodes | 1 (CPU) | 8     | 16     | 32     | 64     | 128    | 256     | 512     | 1024    | 512 (GPU) | 1024    | 2048    | 4096    | Notes |
|-----------------|---------------|---------|-------|--------|--------|--------|--------|---------|---------|---------|-----------|---------|---------|---------|-------|
| EmbeddingBag    | (0,0,0)       | 1.681   | 1.996 | 2.210  | 4.046  | 6.388  | 11.334 | 20.628* | 41.170* | 64.978  | 2.703     |
| EmbeddingBag    | (400,400,400) | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  | * is fastest
| QR EmbeddingBag | (400,400,400) | 5.710   | 8.330 | 10.860 | 17.814 | 31.444 | 61.628 | 112.410 | 217.246 | 422.336*| 9.995     | 12.156  | 15.940  | 31.046  |
| QR EmbeddingBag | (400,400,400) | 4.584   | 5.796 | 8.068  | 14.502 | 25.028 | 46.712 | 90.102  | 178.260*| 364.044 | 8.475     | 10.706  | 13.831  | 20.780  |

#### Quantization
| Embedding       | Quantization  | 1 (CPU) | 16    | 32     | 64     | 128    | 256      | 512     | 1024    | Notes
|-----------------|---------------|---------|-------|--------|--------|--------|----------|---------|---------|
| EmbeddingBag    | None          | 2.396   | 3.463 |        | 44.724 | 86.406 | 167.728  | 331.486*| 680.618 |
| EmbeddingBag    | Dynamic       | 2.376   | 3.516 | 5.858  | 9.132  | 15.876 | 27.849   | 54.623  | 118.678 | room was hot, measure again, average 5 times?
| EmbeddingBag    | Static        | 5.462   | 5.934 | 7.844  | 11.139 | 15.902 | 26.946   | 48.102  | 98.914  | same?
| EmbeddingBag    | QAT           | 6.057   | 8.724 |        | 14.047 | 18.672 | 32.102   | 61.676  | 144.016 |

#### Knowledge Distillation
| Embedding       |  # Deep Nodes | 1 (CPU) | 8     | 16     | 32     | 64     | 128    | 256     | 512     | 1024    | 512 (GPU) | 1024    | 2048    | 4096    | Notes |
|-----------------|---------------|---------|-------|--------|--------|--------|--------|---------|---------|---------|-----------|---------|---------|---------|-------|
| EmbeddingBag    | (400,400,400) | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  | * is fastest
| EmbeddingBag    | (128,128,128) | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  | 
| EmbeddingBag    | (128,128,128) | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  | 
| EmbeddingBag    | (64,64,64)    | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  |
| EmbeddingBag    | (64,64,64)    | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  |
| EmbeddingBag    | (32,32,32)    | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  |
| EmbeddingBag    | (32,32,32)    | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  |
| EmbeddingBag    | (16,16,16)    | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  |
| EmbeddingBag    | (16,16,16)    | 2.570   | 9.134 | 13.120 | 23.550 | 44.724 | 86.406 | 167.728 | 331.486*| 680.618 | 8.951     | 11.251  | 17.433  | 66.624  |
 
#### Ensembles
| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Notes |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|--------|
| QR EmbeddingBag | Dynamic       | (400,400,400) | 0.4468    | 0.8062 | 0.6121    | 21.80 | Nan           |  15.789   | 4 coll
| QR EmbeddingBag | Static        | (400,400,400) | 0.4468    | 0.8061 | 0.6120    | 21.79 | Nan           |  15.777   | 4 coll
| QR EmbeddingBag | Dynamic       | (400,400,400) | 0.4489    | 0.8039 | 0.6085    | 21.44 | Nan           |   7.655   | 16 coll
| QR EmbeddingBag | Static        | (400,400,400) | 0.4489    | 0.8038 | 0.6083    | 21.42 | Nan           |   7.643   | 16 coll


<!-- | QR EmbeddingBag | None          | (64,64,64)    | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | QR + KD
| EmbeddingBag    | Dynamic       | (64,64,64)    | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | KD + Quantization
| QR EmbeddingBag | Dynamic       | (64,64,64)    | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | QR + KD + Quantization -->

| Embedding       | Quantization  | # Deep Nodes  | 1 (CPU) | 16    | 32     | 64     | 128    | 256      | 512     | 1024    | Notes |
|-----------------|---------------|---------------|---------|-------|--------|--------|--------|----------|---------|---------|-------|
| QR EmbeddingBag | Dynamic       | (400,400,400) | 5.180   | 6.494 | 8.038  | 11.406 | 16.702 | 37.160   | 58.712 *| 121.752 | 4 
| QR EmbeddingBag | Static        | (400,400,400) | 5.166   | 6.770 | 8.414  | 11.386 | 16.178 | 28.106   | 50.376 *| 103.644 | 4
| QR EmbeddingBag | Dynamic       | (400,400,400) | 4.178   | 5.082 | 7.232  | 10.446 | 16.482 | 30.586   | 56.124 *| 119.338 | 16
| QR EmbeddingBag | Static        | (400,400,400) | 5.516   | 6.318 | 8.128  | 11.036 | 16.990 | 29.040   | 52.546 *| 109.302 | 16

#### Full Runs
- Epochs: 30
- Early stopping

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Notes |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|--------|
| EmbeddingBag    | None          | (0,0,0)       | 0.4474    | 0.8056 | 0.6104    | 21.69 | 11,956,823    |  47.84    | no overfitting     
| EmbeddingBag    | None          | (400,400,400) | 0.4450    | 0.8082 | 0.6152    | 22.11 | 12,436,824    |  49.780   | early stopped: epoch 14



### Twitter Dataset

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) | Notes      |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|------------|
| Embedding       | None          | (400,400,400) | 0.3173    | 0.9365 |  0.9026   | 53.29 | 62,390,100    |  249.596  | Like, 1 Epoch...then OF
| EmbeddingBag    | None          | (400,400,400) | 0.1018    | 0.8407 |  0.1279   | 14.38 | 62,390,100    |  249.596  | Reply, 1 Epoch
| EmbeddingBag    | None          | (400,400,400) | 0.2275    | 0.8641 |  0.4777   | 29.12 | 62,390,100    |  249.596  | Retweet, 1 Epoch
| EmbeddingBag    | None          | (400,400,400) | 0.0385    | 0.8053 |  0.0280   |  5.18 | 62,390,100    |  249.596  | Retweet with comment, 1 Epoch
| EmbeddingBag    | None          | (400,400,400) | 0.0372    | 0.8069 |  0.0293   |  8.49 | 62,390,100    |  249.596  | Retweet with comment, 6 Epochs, then OF


## References
- https://github.com/rixwew/pytorch-fm
- https://github.com/WayneDW/DeepLight_Deep-Lightweight-Feature-Interactions
- https://github.com/peterliht/knowledge-distillation-pytorch
- https://github.com/rapidsai/deeplearning
- https://github.com/facebookresearch/dlrm
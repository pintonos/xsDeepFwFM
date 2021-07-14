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

| Model   | LogLoss   | AUC    | # Parameters  |
|---------|-----------|--------|---------------|
| LR      | 0.4614    | 0.7899 |  1,086,811    | 
| FM      | 0.4555    | 0.7971 | 11,954,911    | 
| DeepFM  | 0.4475    | 0.8056 | 12,434,912    |
| xDeepFM | 0.4454    | 0.8078 | 12,434,912    |

| Model   | 1 (CPU) | 64        | 128       | 256     | 512      | 512 (GPU) | 1024    | 2048    | 4096    | Notes |
|---------|---------|-----------|-----------|---------|----------|-----------|---------|---------|---------|-------|
| LR      | 0.046   | 0.050     | 0.078     | 0.116   | 0.178    | 0.285     | 0.320   | 0.353   | 0.419   |
| FM      | 0.398   | 0.640     | 0.770     | 1.138   | 2.018    | 2.130     | 2.319   | 2.498   | 2.604   |
| DeepFM  | 1.062   | 20.000    | 38.186    | 72.644  | 142.776  | 4.215     | 3.806   | 4.646   | 5.914   |
| xDeepFM | 14.964  | 1267.092  | 2531.474  | 5011.780| 10080.782| 20.278    | 29.985  | 60.795  | 119.035 |


| Embedding | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Notes |
|-----------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|--------|
| -         | -             | (0,0,0)       | 0.4473    | 0.8058 | 0.6106    | 21.71 | 11,956,823    |  47.84    |
| -         | -             | (400,400,400) | 0.4446    | 0.8086 | 0.6159    | 22.18 | 12,436,824    |  49.780   |
| QR        | -             | (400,400,400) | 0.4452    | 0.8080 | 0.6150    | 22.08 |  7,008,374    |  28.073   | 2 collisions 
| QR        | -             | (400,400,400) | 0.4460    | 0.8069 | 0.6139    | 21.93 |  4,294,354    |  17.217   | 4 collisions
| QR        | -             | (400,400,400) | 0.4496    | 0.8031 | 0.6076    | 21.31 |  1,771,504    |   7.125   | 60 collisions
| -         | Dynamic       | (400,400,400) | 0.4446    | 0.8086 | 0.6159    | 22.17 | 12,436,824    |  48.35    | 
| -         | Static        | (400,400,400) | 0.4448    | 0.8085 | 0.6157    | 22.16 | 12,436,824    |  24.46    | 
| -         | QAT           | (400,400,400) | 0.4459    | 0.8073 | 0.6135    | 21.94 | 12,436,824    |  24.46    |        
| -         | -             | (200,200,200) | 0.4446    | 0.8086 | 0.6160    | 22.18 | 11,028,101    |  44.138   | KD, a=0.1, t=3
| -         | -             | (200,200,200) | 0.4449    | 0.8083 | 0.6154    | 22.13 | 11,028,101    |  44.138   | no KD
| -         | -             | (100,100,100) | 0.4450    | 0.8082 | 0.6152    | 22.11 | 10,928,101    |  43.736   | KD, a=0.1, t=3
| -         | -             | (100,100,100) | 0.4453    | 0.8078 | 0.6146    | 22.06 | 10,928,101    |  43.736   | no KD TODO


#### Embeddings - Latency

| Embedding       |  # Deep Nodes | 1 (CPU) | 64     | 128    | 256     | 512     | 512 (GPU) | 1024    | 2048    | 4096    | Notes |
|-----------------|---------------|---------|--------|--------|---------|---------|-----------|---------|---------|---------|-------|
| Embedding       | (0,0,0)       | 1.681   | 5.788  |  9.834 | 18.228  | 36.250  | 5.103     | 6.657   |  9.867  | 15.849  |
| Embedding       | (400,400,400) | 3.652   | 72.418 |143.632 | 279.482 | 549.400 | 7.029     | 7.806   | 11.757  | 18.015  |
| QR EmbeddingBag | (400,400,400) | 2.482   | 76.736 |151.032 | 301.912 | 609.880 | 6.772     | 9.772   | 14.440  | 24.746  | 2 collisions TODO measure again 
| QR EmbeddingBag | (400,400,400) | 4.142   | 67.244 |129.276 | 253.910 | 504.246 | 9.995     | 10.356  | 14.040  | 20.046  | 4 collisions
| QR EmbeddingBag | (400,400,400) | 4.284   | 39.628 | 75.712 | 147.116 | 288.920 | 9.275     | 10.506  | 14.431  | 21.580  | 60 collisions

#### Quantization - Latency

| Quantization  | 1 (CPU) | 64     | 128    | 256      | 512     |
|---------------|---------|--------|--------|----------|---------|
| None          | 3.652   | 72.418 |143.632 | 279.482  | 549.400 |
| Dynamic       | 2.876   | 9.220  | 14.284 | 25.728   | 49.082  | 
| Static        | 5.062   | 10.139 | 16.034 | 27.146   | 49.352  |
| QAT           | 5.457   | 11.141 | 16.404 | 26.892   | 46.676  |

#### Knowledge Distillation - Latency

| Embedding       |  # Deep Nodes | 1 (CPU) | 64     | 128    | 256     | 512     | 512 (GPU) | 1024    | 2048    | 4096    |
|-----------------|---------------|---------|--------|--------|---------|---------|-----------|---------|---------|---------|
| EmbeddingBag    | (400,400,400) | 3.652   | 72.418 |143.632 | 279.482 | 549.400 | 7.029     | 7.806   | 11.757  | 18.015  | 
| EmbeddingBag    | (200,200,200) | 0.618   | 10.626 | 18.752 |  37.428 |  73.060 | 2.489     | 2.803   | 3.175   |  3.886  | 
| EmbeddingBag    | (100,100,100) | 0.536   |  2.812 |  4.818 |   8.454 |  16.022 | 2.520     | 2.817   | 3.094   |  3.325  | 

 
#### Ensembles

| Embedding     | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Notes |
|---------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|--------|
| QR Embedding  | None          | (200,200,200) | 0.4449    | 0.8083 | 0.6155    | 22.13 |  5,599,651    |  22.431   | KD + 2 coll
| QR Embedding  | None          | (200,200,200) | 0.4455    | 0.8076 | 0.6143    | 22.02 |  2,885,631    |  11.575   | KD + 4 coll
| QR Embedding  | None          | (100,100,100) | 0.4459    | 0.8072 | 0.6138    | 21.96 |  2,785,631    |  11.172   | KD + 4 coll
| Embedding     | Static        | (200,200,200) | 0.4448    | 0.8082 | 0.6159    | 22.15 | 11,028,101    |  19.781   | KD
| Embedding     | Static        | (100,100,100) | 0.4452    | 0.8078 | 0.6151    | 22.07 | 10,928,101    |  19.671   | KD


#### Ensembles - Latency

| Embedding    | Quantization  | # Deep Nodes  | 1 (CPU) | 64     | 128    | 256      | 512     | Notes |
|--------------|---------------|---------------|---------|--------|--------|----------|---------|-------|
| QR Embedding | None          | (200,200,200) | 1.486   | 10.458 | 19.296 | 37.062   | 72.264  | KD + 4 coll
| QR Embedding | None          | (100,100,100) | 1.260   |  3.058 |  4.778 |  8.518   | 16.136  | KD + 4 coll
| Embedding    | Static        | (200,200,200) | 6.132   |  6.826 |  6.772 |  7.134   |  9.414  | KD
| Embedding    | Static        | (100,100,100) | 5.432   |  5.726 |  6.120 |  6.290   |  7.240  | KD

### Twitter
- Threshold: 15
- Epochs: 50 with early stopping
- Dropout: 0.2

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) | Notes      |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|------------|
| EmbeddingBag    | None          | (0,0,0)       | 0.3248    | 0.9368 |  0.9049   | 52.19 | 131,425,764   |  525.721  | Like
| EmbeddingBag    | None          | (0,0,0)       | 0.2202    | 0.8754 |  0.5223   | 31.35 | 131,425,764   |  525.721  | Retweet
| EmbeddingBag    | None          | (0,0,0)       | 0.0988    | 0.8463 |  0.1388   | 16.90 | 131,425,764   |  525.721  | Reply
| EmbeddingBag    | None          | (0,0,0)       | 0.0360    | 0.8231 |  0.0456   | 11.43 | 131,425,764   |  525.721  | Retweet with comment
| EmbeddingBag    | None          | (400,400,400) | 0.3180    | 0.9398 |  0.9100   | 53.19 | 131,937,765   |  527.786  | Like
| EmbeddingBag    | None          | (400,400,400) | 0.2161    | 0.8807 |  0.5355   | 32.63 | 131,937,765   |  527.786  | Retweet
| EmbeddingBag    | None          | (400,400,400) | 0.0961    | 0.8549 |  0.1525   | 19.17 | 131,937,765   |  527.786  | Reply
| EmbeddingBag    | None          | (400,400,400) | 0.0356    | 0.8122 |  0.0358   | 11.75 | 131,937,765   |  527.786  | Retweet with comment 


## References
- https://github.com/rixwew/pytorch-fm
- https://github.com/WayneDW/DeepLight_Deep-Lightweight-Feature-Interactions
- https://github.com/peterliht/knowledge-distillation-pytorch
- https://github.com/rapidsai/deeplearning
- https://github.com/facebookresearch/dlrm
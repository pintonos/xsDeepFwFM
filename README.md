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

| Model   | Embedding        | LogLoss   | AUC    | # Parameters  |
|---------|------------------|-----------|--------|---------------|
| LR      |                  | 0.4614    | 0.7899 |  1,086,811    |
| FM      |                  | 0.4555    | 0.7971 | 11,954,911    | TODO
| DeepFM  |                  | 0.4475    | 0.8056 | 12,434,912    |
| xDeepFM |                  | 0.4475    | 0.8056 | 12,434,912    | TODO GCP

| Embedding | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Notes |
|-----------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|--------|
| -         | -             | (0,0,0)       | 0.4473    | 0.8058 | 0.6106    | 21.71 | 11,956,823    |  47.84    |
| -         | -             | (400,400,400) | 0.4446    | 0.8086 | 0.6159    | 22.18 | 12,436,824    |  49.780   | TODO!!!!
| QR        | -             | (400,400,400) | 0.4458    | 0.8072 | 0.6140    | 21.97 |  4,294,354    |  17.217   | 4 collisions, TODO train more
| QR        | -             | (400,400,400) | 0.4496    | 0.8031 | 0.6076    | 21.31 |  1,771,504    |   7.125   | 60 collisions
| -         | Dynamic       | (400,400,400) | 0.4446    | 0.8086 | 0.6159    | 22.17 | 12,436,824    |  48.35    | 
| -         | Static        | (400,400,400) | 0.4448    | 0.8085 | 0.6157    | 22.16 | 12,436,824    |  24.46    | 
| -         | QAT           | (400,400,400) | 0.4459    | 0.8073 | 0.6135    | 21.94 | 12,436,824    |  24.46    |        
| -         | -             | (200,200,200) | 0.4448    | 0.8084 | 0.6156    | 22.15 | 11,028,101    |  44.138   | KD, a=0.05, t=3 TODO train more
| -         | -             | (200,200,200) | 0.4449    | 0.8082 | 0.6154    | 22.13 | 11,028,101    |  44.138   | KD, a=0.1, t=3  
| -         | -             | (200,200,200) | 0.4449    | 0.8083 | 0.6154    | 22.13 | 11,028,101    |  44.138   | no KD correct? TODO train more
| -         | -             | (100,100,100) | 0.4449    | 0.8082 | 0.6154    | 22.13 | 11,028,101    |  44.138   | KD, a=0.1, t=3  TODO
| -         | -             | (100,100,100) | 0.4449    | 0.8082 | 0.6154    | 22.13 | 11,028,101    |  44.138   | KD, a=0.05, t=3  TODO 
| -         | -             | (100,100,100) | 0.4453    | 0.8078 | 0.6146    | 22.06 | 11,028,101    |  44.138   | no KD


2021-06-28 20:09:02 - INFO - test loss: 0.445213 auc: 0.808072 prauc: 0.6150 rce: 22.0874
2021-06-28 20:09:02 - INFO - epoch: 5

#### Embeddings - Latency

| Embedding       |  # Deep Nodes | 1 (CPU) | 64     | 128    | 256     | 512     | 512 (GPU) | 1024    | 2048    | 4096    | Notes |
|-----------------|---------------|---------|--------|--------|---------|---------|-----------|---------|---------|---------|-------|
| EmbeddingBag    | (0,0,0)       | 1.681   | 5.788  |  9.834 | 18.228  | 36.250  | 5.103     | 6.657   |  9.867  | 15.849  |
| EmbeddingBag    | (400,400,400) | 3.652   | 72.418 |143.632 | 279.482 | 549.400 | 7.029     | 7.806   | 11.757  | 18.015  | 
| QR EmbeddingBag | (400,400,400) | 4.142   | 67.244 |129.276 | 253.910 | 504.246 | 9.995     | 10.356  | 14.040  | 20.046  | 4 collisions
| QR EmbeddingBag | (400,400,400) | 4.284   | 39.628 | 75.712 | 147.116 | 288.920 | 9.275     | 10.506  | 14.431  | 21.580  | 60 collisions

#### Quantization - Latency
| Embedding       | Quantization  | 1 (CPU) | 64     | 128    | 256      | 512     | Notes |
|-----------------|---------------|---------|--------|--------|----------|---------|-------|
| EmbeddingBag    | None          | 3.652   | 72.418 |143.632 | 279.482  | 549.400 |
| EmbeddingBag    | Dynamic       | 2.876   | 9.220  | 14.284 | 25.728   | 49.082  | 
| EmbeddingBag    | Static        | 5.062   | 10.139 | 16.034 | 27.146   | 49.352  |
| EmbeddingBag    | QAT           | 5.457   | 11.141 | 16.404 | 26.892   | 46.676  |

#### Knowledge Distillation - Latency
- TODO

| Embedding       |  # Deep Nodes | 1 (CPU) | 64     | 128    | 256     | 512     | 512 (GPU) | 1024    | 2048    | 4096    | Notes |
|-----------------|---------------|---------|--------|--------|---------|---------|-----------|---------|---------|---------|-------|
| EmbeddingBag    | (400,400,400) | 3.652   | 72.418 |143.632 | 279.482 | 549.400 | 7.029     | 7.806   | 11.757  | 18.015  | 
| EmbeddingBag    | (200,200,200) | 0.816   | 10.626 | 19.952 |  39.928 |  81.148 | 3.992     | 2.973   | 3.285   |  3.806  | 
| EmbeddingBag    | (100,100,100) | 0.764   | 4.030  |  6.918 |  11.454 |  21.822 | 5.820     | 3.217   | 3.294   |  3.825  | 

 
#### Ensembles
- TODO

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Notes |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|--------|
| QR EmbeddingBag | Dynamic       | (400,400,400) | 0.4468    | 0.8062 | 0.6121    | 21.80 | Nan           |  15.789   | 4 coll
| QR EmbeddingBag | Static        | (400,400,400) | 0.4468    | 0.8061 | 0.6120    | 21.79 | Nan           |  15.777   | 4 coll
| QR EmbeddingBag | Dynamic       | (100,100,100) | 0.4468    | 0.8062 | 0.6121    | 21.80 | Nan           |  15.789   | 4 coll + KD


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
| EmbeddingBag    | None          | (400,400,400) | 0.0356    | 0.8122 |  0.0358   | 11.75 | 131,937,765   |  527.786  | Retweet with comment 2048 


## References
- https://github.com/rixwew/pytorch-fm
- https://github.com/WayneDW/DeepLight_Deep-Lightweight-Feature-Interactions
- https://github.com/peterliht/knowledge-distillation-pytorch
- https://github.com/rapidsai/deeplearning
- https://github.com/facebookresearch/dlrm
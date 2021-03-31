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
| EmbeddingBag    | None          | (128,128,128) | 0.4461    | 0.8070 | 0.6131    | 21.91 | ----------    |  ------   | KD 
| EmbeddingBag    | None          | (128,128,128) | 0.4459    | 0.8072 | 0.6135    | 22.95 | ----------    |  ------   | without KD 
| EmbeddingBag    | None          | (64,64,64)    | 0.4466    | 0.8065 | 0.6121    | 21.83 | 11,956,822    |  47.842   | KD
| EmbeddingBag    | None          | (64,64,64)    | 0.4467    | 0.8064 | 0.6121    | 21.81 | 11,956,822    |  47.842   | without KD 
| EmbeddingBag    | None          | (32,32,32)    | 0.4470    | 0.8061 | 0.6112    | 21.75 | 11,971,672    |  47.911   | KD * TODO more parameters? why?
| EmbeddingBag    | None          | (32,32,32)    | 0.4470    | 0.8060 | 0.6112    | 21.75 | 11,971,672    |  47.842   | without KD 
| EmbeddingBag    | None          | (16,16,16)    | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | KD 
| EmbeddingBag    | None          | (16,16,16)    | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | without KD 

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
| QR EmbeddingBag | Dynamic       | (64,64,64) | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | QR + Quantization
| QR EmbeddingBag | None          | (64,64,64) | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | QR + KD
| EmbeddingBag    | Dynamic       | (64,64,64) | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | KD + Quantization
| QR EmbeddingBag | Dynamic       | (64,64,64) | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | QR + KD + Quantization


### Twitter Dataset

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) | Notes
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|------------|
| Embedding       | None          | (400,400,400) | 0.3173    | 0.9365 |  0.9026   | 53.29 | 62,390,64    |  249.596  | Like, 1 Epoch
| EmbeddingBag    | None          | (400,400,400) | 0.1018    | 0.8407 |  0.1279   | 14.38 | 62,390,64    |  249.596  | Reply, 1 Epoch


## References
- https://github.com/rixwew/pytorch-fm
- https://github.com/WayneDW/DeepLight_Deep-Lightweight-Feature-Interactions
- https://github.com/peterliht/knowledge-distillation-pytorch
- https://github.com/rapidsai/deeplearning
- https://github.com/facebookresearch/dlrm
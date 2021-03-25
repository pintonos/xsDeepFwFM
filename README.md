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
- Epochs: 1
- Batch Size (Training): 2048
- Intel Xeon E3-1231v3 & NVIDIA GTX 970

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Notes |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|--------|
| Embedding       | None          | (0,0,0)       | 0.4499    | 0.8030 | 0.6060    | 21.26 | 11,956,823    |  47.842   | 
| EmbeddingBag    | None          | (0,0,0)       | 0.4498    | 0.8030 | 0.6061    | 21.27 | 11,956,823    |  47.842   | 
| Embedding       | None          | (400,400,400) | 0.4479    | 0.8050 | 0.6099    | 21.60 | 12,436,824    |  49.780   |
| EmbeddingBag    | None          | (400,400,400) | 0.4478    | 0.8051 | 0.6103    | 21.63 | 12,436,824    |  49.780   | 
| QR EmbeddingBag | None          | (400,400,400) | 0.4517    | 0.8007 | 0.6034    | 20.93 |  4,294,354    |  17.216   | 4 collisions
| QR EmbeddingBag | None          | (400,400,400) | 0.4538    | 0.7981 | 0.5997    | 20.56 |  2,260,814    |   9.082   | 16 collisions            
| EmbeddingBag    | Dynamic       | (400,400,400) | 0.4478    | 0.8051 | 0.6102    | 21.62 | 11,959,223    |  48.35    |  
| EmbeddingBag    | Static        | (400,400,400) | 0.4478    | 0.8051 | 0.6102    | 21.62 | NaN           |  24.46    | 
| EmbeddingBag    | QAT           | (400,400,400) | 0.4409    | 0.8108 | 0.6167    | 22.53 | NaN           |  7.757    | *
| EmbeddingBag    | None          | (100,100,100) | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | KD *
| EmbeddingBag    | None          | (100,100,100) | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | without KD *


| Embedding       |  # Deep Nodes | 1 (CPU) | 16 | 128   | 256   | 512 | 512 (GPU) | 1024 | 2048 | 4096 |
|-----------------|---------------|---------|----|-------|-------|-----|-----------|------|------|------|
| EmbeddingBag    | (0,0,0)       |
| EmbeddingBag    | (400,400,400  |

| Embedding       | Quantization  | 1 (CPU) | 16    | 128    | 256    | 512    |
|-----------------|---------------|---------|-------|--------|--------|--------|
| EmbeddingBag    | Dynamic       | 2.677   | 3.531 | 15.529 | 28.710 | 55.872
| EmbeddingBag    | Static        |
| EmbeddingBag    | QAT           |
 
#### Ensembles
| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Notes |
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|--------|
| QR EmbeddingBag | None          | (100,100,100) | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | QR + KD
| EmbeddingBag    | Static        | (100,100,100) | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | KD + Quantization
| QR EmbeddingBag | Dynamic       | (100,100,100) | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | QR + KD + Quantization
| QR EmbeddingBag | Static        | (100,100,100) | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | QR + KD + Quantization
| QR EmbeddingBag | QAT           | (100,100,100) | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | QR + KD + Quantization


### Twitter Dataset

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Time per batch (1-Threads)(ms)  | Time per item (1-Threads)(ms)  |   Time per batch (CUDA)(ms)  | Time per item (CUDA)(ms) | Notes
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|----------------------------------|--------------------------------|------------------------------|--------------------------|------------|
| Embedding       | None          | (400,400,400) | 0.3173    | 0.9365 |  0.9026   | 53.29 | 62,390,100    |  249.596  | NaN                              | NaN                            | NaN                          | Nan                      | Like, 3,5h per Epoch 


## References
- https://github.com/rixwew/pytorch-fm
- https://github.com/WayneDW/DeepLight_Deep-Lightweight-Feature-Interactions
- https://github.com/peterliht/knowledge-distillation-pytorch
- https://github.com/rapidsai/deeplearning
- https://github.com/facebookresearch/dlrm
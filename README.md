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
- Epochs: 5
- Batch Size: 2048
- 3 Deep Layers
- Intel Xeon E3-1231v3 & NVIDIA GTX 970

| Embedding       | Quantization  | # Deep Nodes  | LogLoss   | AUC    | PRAUC     | RCE   | # Parameters  | Size (MB) |  Time per batch (1-Threads)(ms)  | Time per item (1-Threads)(ms)  |   Time per batch (CUDA)(ms)  | Time per item (CUDA)(ms) | Notes
|-----------------|---------------|---------------|-----------|--------|-----------|-------|---------------|-----------|----------------------------------|--------------------------------|------------------------------|--------------------------|------------|
| Embedding       | None          | (0,0,0)       | 0.4424    | 0.8091 | 0.6131    | 22.24 | 11,956,822    |  47.842   | 168.089                          | 0.08207                        | 7.230                        | 0.00353                  |Epochs: 10
| EmbeddingBag    | None          | (0,0,0)       | 0.4426    | 0.8090 | 0.6127    | 22.21 | 11,956,822    |  47.842   | 170.367                          | 0.08319                        | 9.368                        | 0.00457                  |Epochs: 10
| Embedding       | None          | (400,400,400) | 0.4551    | 0.7872 | 0.5620    | 18.65 | 4,479,651     | 17.935    | 1.979                            | 1299.191                       | NaN                          | NaN                      | 
| EmbeddingBag    | None          | (400,400,400) | 0.4404    | 0.8117 | 0.6183    | 22.64 | 12,436,823    |  49.78    | 1346.075                         | 0.65726                        | 11.157                       | 0.00545                  | 
| QR EmbeddingBag | None          | (400,400,400) | 0.4573    | 0.7844 | 0.5750    | 19.38 | 1,481,681     |  5.949    | 5.062                            | 1476.571                       | NaN                          | NaN                      |                        
| EmbeddingBag    | Dynamic       | (400,400,400) | 0.4353    | 0.8169 | 0.6260    | 23.50 | 11,959,222    |  48.35    | 272.060                          | 0.13284                        | NaN                          | NaN                      | 
| EmbeddingBag    | Static        | (400,400,400) | 0.4353    | 0.8169 | 0.6260    | 23.49 | NaN           |  24.46    | 256.301                          | 0.12515                        | NaN                          | NaN                      | 
| EmbeddingBag    | QAT           | (400,400,400) | 0.4409    | 0.8108 | 0.6167    | 22.53 | NaN           |  7.757    | NaN                              | NaN                            | NaN                          | NaN                      |Epochs: 10



## References
- https://github.com/rixwew/pytorch-fm
- https://github.com/WayneDW/DeepLight_Deep-Lightweight-Feature-Interactions
- https://github.com/peterliht/knowledge-distillation-pytorch
- https://github.com/rapidsai/deeplearning
- https://github.com/facebookresearch/dlrm

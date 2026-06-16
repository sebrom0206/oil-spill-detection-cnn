# Oil Spill Detection in SAR Imagery using CNNs

Binary classification of oil spill features in Sentinel-1 SAR 
satellite imagery. Developed as part of the Machine Learning & 
Deep Learning course at Copenhagen Business School.

## Results

| Model | F1 Macro | F1 Oil Spill | AUC-ROC |
|-------|----------|--------------|---------|
| CNN Res SE | **0.944** | **0.926** | 0.983 |
| EfficientNetB0 (baseline) | 0.936 | 0.917 | 0.989 |
| CNN Res | 0.930 | 0.907 | 0.978 |
| CNN Conv | 0.823 | 0.783 | 0.938 |

CNN Res SE surpasses the pretrained EfficientNetB0 baseline 
on F1 with ~4x fewer parameters.

## Dataset
5,630 Sentinel-1 SAR image chips from Australian waters 
(Kaggle: sentinel-1-sar-oil-spill-detection-dataset).

## Techniques
- Custom CNN architectures of increasing complexity
- Squeeze-and-Excitation (SE) attention blocks
- Class weighting for imbalanced data
- Grad-CAM for model interpretability

## Tech Stack
Python · TensorFlow · Keras · NumPy · scikit-learn · Matplotlib

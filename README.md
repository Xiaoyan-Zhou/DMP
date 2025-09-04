# DMP
## Uncertainty Aware Dynamic Multi-Prototype Representation for Out-of-Distribution Detection in SAR Imagery

## Abstract
Out-of-distribution (OOD) detection is essential for reliable automatic target recognition (ATR) in open-world scenarios. Although widely studied in optical imagery, research on OOD detection in synthetic aperture radar (SAR) images remains limited. SAR imagery poses unique challenges due to high intra-class variability and low inter-class separability, which existing distance-based methods with simplistic class representations fail to handle. To address these issues, we propose two dynamic multi-prototype representation methods: Dynamic Multi-Prototype Clustering (DMPC) and Uncertainty-Aware Dynamic Multi-Prototype Learning (DMPL). DMPC is a post-hoc method that applies hierarchical clustering to assign adaptive prototypes per class, enabling accurate distance-based OOD scoring without retraining. In contrast, DMPL is an end-to-end framework that enhances discrimination in SAR imagery by improving intra-class compactness and inter-class separability. It further incorporates an uncertainty-aware filter that updates prototypes only with low intra-class uncertainty samples, ensuring stable and diverse representations. Extensive experiments on benchmark SAR-OOD datasets show that the proposed methods outperform state-of-the-art baselines, demonstrating the effectiveness of dynamic multi-prototype representation for OOD detection in SAR imagery.

## Usage
### train

```sh
python python 0-main.py --filter --uncertainty_th 0.9 --loss_init HierarchicalClustering --K 40 --batch_size 32 --lr_model 0.01 --lr_pro 0.5 --centroids_path ./init_model_npy/resnet18_CE_10_K40_HierarchicalClustering_centroids.npy --label_path ./init_model_npy/resnet18_CE_10_K40_HierarchicalClustering_labels.npy
```
## Citation
If you use this code for SAR ATR, please cite the following work:
```

```


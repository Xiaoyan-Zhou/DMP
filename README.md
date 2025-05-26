# DMP
## Uncertainty Aware Dynamic Multi-Prototype Representation for Out-of-Distribution Detection in SAR Imagery

## Abstract
Out-of-distribution (OOD) detection is vital for robust automatic target recognition (ATR) in open-world scenarios. While OOD detection has been extensively explored in the optical image domain, studies focusing on synthetic aperture radar (SAR) image remain limited. SAR image poses unique challenges of high intra-class variability and low inter-class separability due to its imaging mechanism. Moreover, existing distance-based OOD methods often rely on simplistic class representations, failing to capture the complex and heterogeneous feature distributions inherent to SAR images. To address these issues, we propose two dynamic multi-prototype representation methods: Dynamic Multi-Prototype Clustering (DMPC) and Dynamic Multi-Prototype Learning (DMPL). DMPC is a post-hoc method that employs hierarchical clustering to assign an adaptive number of prototypes per class, enabling more accurate distance-based OOD scoring without retraining. DMPL is an end-to-end framework that enhances intra-class compactness and inter-class separability, employing an uncertainty-aware filter that updates prototypes only with low intra-class uncertainty samples for stability and diversity prototype representation. Extensive experiments on benchmark SAR-OOD datasets demonstrate that the proposed methods consistently outperform state-of-the-art baselines, highlighting the effectiveness of dynamic multi-prototype representation in SAR-based recognition tasks.

## Usage
### train

```sh
python python 0-main.py --filter --uncertainty_th 0.9 --loss_init HierarchicalClustering --K 40 --batch_size 32 --lr_model 0.01 --lr_pro 0.5 --centroids_path ./init_model_npy/resnet18_CE_10_K40_HierarchicalClustering_centroids.npy --label_path ./init_model_npy/resnet18_CE_10_K40_HierarchicalClustering_labels.npy
```
## Citation
If you use this code for SAR ATR, please cite the following work:
```

```


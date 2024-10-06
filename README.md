# DMP
## Dynamic Multi-Prototype Representation with Uncertainty Filter for SAR Out-of-Distribution Detection

## This paper will be submitted to ISPRS

## Abstract
Out-of-distribution (OOD) detection is crucial for ensuring the reliability of models in open-world environments. Although extensive research has been conducted on OOD detection algorithms in the optical image processing domain, systematic studies in the field of synthetic aperture radar (SAR) remain limited. Existing methods often adapt optical image OOD detection algorithms without accounting for the unique characteristics of SAR images, specifically their small inter-class variations and large intra-class variations. To address this gap, we propose two dynamic multi-prototype representation approaches—Dynamic Multi-Prototype Clustering (DMPC) and Dynamic Multi-Prototype Learning (DMPL)—designed to capture the intra-class diversity of SAR targets and accurately represent different categories, thereby achieving superior SAR OOD detection. DMPC is a post-hoc method employing hierarchical clustering to create a dynamic number of prototypes per class based on data distribution, revealing the non-globular nature of SAR feature spaces. The prototypes generated serve as initial inputs for DMPL. DMPL is an end-to-end training framework consisting of an In-Distribution Feature Enhancement Module (IFEM) and a Prototype Optimization Module (POM). The IFEM enhances ID data features using a proposed Joint Prototype and Cross-Entropy Loss (JPEL), which emphasizes intra-class aggregation and inter-class separation. The POM introduces a novel uncertainty estimation method based on feature distance and entropy, utilizing an uncertainty filter that updates prototypes only with low intra-class uncertainty samples, thus ensuring prototype stability and diversity. Experiments on SAR-OOD datasets demonstrate that our approaches achieve state-of-the-art results, significantly outperforming existing methods. This work highlights the importance of considering SAR-specific data characteristics in OOD detection and sets a new standard for applying hierarchical clustering and prototype-based learning in this challenging domain. Our code is available on GitHub at https://github.com/Xiaoyan-Zhou/DMP.

## Usage
### train

```sh
python python 0-main.py --filter --uncertainty_th 0.9 --loss_init HierarchicalClustering --K 40 --batch_size 32 --lr_model 0.01 --lr_pro 0.5 --centroids_path ./init_model_npy/resnet18_CE_10_K40_HierarchicalClustering_centroids.npy --label_path ./init_model_npy/resnet18_CE_10_K40_HierarchicalClustering_labels.npy
```
## Citation
If you use this code for SAR ATR, please cite the following work:
```

```


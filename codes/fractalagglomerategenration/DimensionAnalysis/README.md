# Agglomerate Dimension Analysis

This folder contains a simple implementation of the 3D box-counting method proposed by Wang et al. for estimating the fractal dimension of generated agglomerates.

The script expects an aggregate CSV file in the x, y, z, r format and plots the resulting box-counting data together with the fitted linear regression.

> [!CAUTION]  
> Due to the high memory requirements of this 3D box-counting algorithm, this implementation is only practical for relatively small aggregates (up to a few hundred primary particles, depending on the selected grid resolution).

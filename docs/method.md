# CoordNet: Technical Details

This document provides the full mathematical formulation of the CoordNet architecture.

## E(3)-Equivariant Message Passing

CoordNet builds on the PaiNN-style equivariant message passing framework. Each atom $i$ maintains:

- **Scalar features** $\mathbf{s}_i \in \mathbb{R}^F$: Rotation/translation invariant
- **Vector features** $\mathbf{v}_i \in \mathbb{R}^{3 \times F}$: Transform as SO(3) vectors under rotation

### Message Passing Updates

For each edge $(i, j)$ with distance $d_{ij} = \|\mathbf{r}_j - \mathbf{r}_i\|$ and unit direction vector $\hat{\mathbf{r}}_{ij} = (\mathbf{r}_j - \mathbf{r}_i) / d_{ij}$:

**Scalar update:**

$$\Delta \mathbf{s}_i = \sum_{j \in \mathcal{N}(i)} \phi_s(\mathbf{s}_j) \odot W(d_{ij})$$

**Vector update:**

$$\Delta \mathbf{v}_i = \sum_{j \in \mathcal{N}(i)} \left[ \mathbf{m}^{sv}_{ij} + (\mathbf{v}_j \cdot \hat{\mathbf{r}}_{ij}) \mathbf{m}^{vv}_{ij} \right] \otimes \hat{\mathbf{r}}_{ij}$$

Where:
- $\phi_s$: Learnable scalar MLP
- $W(d_{ij})$: Distance-dependent filter (RBF-based)
- $\mathbf{m}^{sv}_{ij}, \mathbf{m}^{vv}_{ij}$: Scalar-to-vector and vector-to-vector message components
- $\odot$: Element-wise (Hadamard) product
- $\otimes$: Outer product (broadcasting scalar/vector to vector features)
- $\mathcal{N}(i)$: Neighbors of atom $i$ within cutoff radius

### Distance Encoding

Interatomic distances are encoded using Gaussian radial basis functions:

$$\text{RBF}(d) = \exp\left( -\frac{(d - \mu_k)^2}{2\sigma^2} \right), \quad k = 1, \ldots, K$$

with $K = 20$ centers uniformly spaced from 0 to $r_{\text{cut}} = 5.0$ Å.

A smooth cosine cutoff ensures continuous derivatives:

$$f_{\text{cut}}(d) = \begin{cases}
\frac{1}{2}\left(1 + \cos\left(\frac{\pi d}{r_{\text{cut}}}\right)\right) & d \leq r_{\text{cut}} \\
0 & d > r_{\text{cut}}
\end{cases}$$

## Dual Embedding

Atomic representations are initialized by combining:

1. **Element embedding**: $\mathbf{h}_Z = \text{Embed}(Z)$ where $Z$ is the atomic number
2. **Charge embedding**: $\mathbf{h}_Q = \text{Embed}(Q)$ where $Q$ is the total molecular charge

$$\mathbf{s}_i^{(0)} = \mathbf{h}_{Z_i} + \mathbf{h}_Q$$

This conditions the representation on the electronic boundary condition (total charge).

## Readout and Prediction Heads

### Energy / Gap (Scalar Invariants)

Molecular-level scalar features are obtained by sum pooling:

$$\mathbf{s}_{\text{mol}} = \sum_{i \in \text{mol}} \mathbf{s}_i$$

Followed by an MLP for property prediction:

$$E = \text{MLP}_E(\mathbf{s}_{\text{mol}}) + \sum_i E_{\text{ref}}(Z_i)$$

### Dipole Moment (Equivariant → Scalar Magnitude)

The dipole vector is computed via equivariant pooling of vector features:

$$\boldsymbol{\mu} = \sum_{i \in \text{mol}} \mathbf{W}_\mu \mathbf{v}_i$$

where $\mathbf{W}_\mu$ is a learnable linear projection. The output is the magnitude:

$$|\boldsymbol{\mu}| = \|\boldsymbol{\mu}\|_2$$

This ensures E(3) equivariance: the dipole vector rotates with the molecule, but its magnitude is invariant.

## References

- Schütt, K. T., et al. "Equivariant message passing for the prediction of tensorial properties and molecular spectra." *ICML 2021*. (PaiNN)
- Gasteiger, J., et al. "GemNet: Universal Directional Graph Neural Networks for Molecules." *NeurIPS 2021*.


# Discriminative Mutual Information Estimators based on the variational representation of the $f$-divergence

PyTorch implementation for the paper "Variational $f$-Divergence and Derangements for Discriminative Mutual Information Estimation".

---

## General description

The code comprises the implementation of various existing mutual information (MI) estimators (e.g. MINE, NWJ, InfoNCE, SMILE, NJEE) that are compared with our proposed new class of MI estimators: $f$-DIME:

$I_{fDIME}(X;Y) = \mathbb{E}_ {p_{XY}(\mathbf{x},\mathbf{y})} \biggl[ \log \biggl( \bigl( f^* \bigr)^ {'} \bigl(\hat{T}(\mathbf{x},\mathbf{y})\bigr) \biggr) \biggr]$, 

where $\hat{T}$ is obtained by maximizing

$\mathcal{J}_ {f}(T) =  \mathbb{E}_ {(\mathbf{x},\mathbf{y}) \sim p_ {XY}(\mathbf{x},\mathbf{y})}\biggl[T\bigl(\mathbf{x},\mathbf{y}\bigr)-f^*\biggl(T\bigl(\mathbf{x},\sigma(\mathbf{y})\bigr)\biggr)\biggr]$.

In particular, we developed three different estimators, based on three different $f$-divergences:

- **KL-DIME** (based on the Kullback-Leibler divergence)

   $I_ {KL-DIME}(X;Y) :=  \mathbb{E}_ {(\mathbf{x},\mathbf{y}) \sim p_ {XY}(\mathbf{x},\mathbf{y})}\biggl[ \log \biggl(\hat{D}(\mathbf{x},\mathbf{y})\biggr) \biggr]$,

   where $\hat{D}$ is obtained by maximizing  
   $\mathcal{J}_ {KL}(D) = \mathbb{E}_ {(\mathbf{x},\mathbf{y}) \sim p_ {XY}(\mathbf{x},\mathbf{y})}\biggl[\log\bigl(D\bigl(\mathbf{x},\mathbf{y}\bigr)\bigr)\biggr] -\mathbb{E}_ {(\mathbf{x},\mathbf{y}) \sim p_ {X}(\mathbf{x})p_ {Y}(\mathbf{y})}\biggl[D\bigl(\mathbf{x},\mathbf{y}\bigr)\biggr]+1$.

- **HD-DIME** (based on the squared Hellinger distance)

   $I_ {HD-DIME}(X;Y) :=  \mathbb{E}_ {(\mathbf{x},\mathbf{y}) \sim p_ {XY}(\mathbf{x},\mathbf{y})}\biggl[ \log \biggl(\frac{1}{\hat{D}^2(\mathbf{x},\mathbf{y})}\biggr) \biggr]$,

   where $\hat{D}$ is obtained by maximizing  
$\mathcal{J}_ {HD}(D) = 2-\mathbb{E}_ {(\mathbf{x},\mathbf{y}) \sim p_ {XY}(\mathbf{x},\mathbf{y})}\biggl[D\bigl(\mathbf{x},\mathbf{y}\bigr)\biggr] -\mathbb{E}_ {(\mathbf{x},\mathbf{y}) \sim p_{X}(\mathbf{x})p_ {Y}(\mathbf{y})}\biggl[\frac{1}{D(\mathbf{x},\mathbf{y})}\biggr]$.


- **GAN-DIME** (based on the GAN/Jensen-Shannon divergence)

   $I_ {GAN-DIME}(X;Y) :=  \mathbb{E}_ {(\mathbf{x},\mathbf{y}) \sim p_ {XY}(\mathbf{x},\mathbf{y})}\biggl[ \log \biggl(\frac{1-\hat{D}(\mathbf{x},\mathbf{y})}{\hat{D}(\mathbf{x},\mathbf{y})}\biggr) \biggr]$,

   where $\hat{D}$ is obtained by maximizing  
$\mathcal{J}_ {GAN}(D) = \mathbb{E}_ {(\mathbf{x},\mathbf{y}) \sim p_ {XY}(\mathbf{x},\mathbf{y})} \biggl[ \log \bigl( 1- D \bigl( \mathbf{x}, \mathbf{y} \bigr) \bigr) \biggr] +\mathbb{E}_ {(\mathbf{x},\mathbf{y}) \sim p_{X}(\mathbf{x})p_ {Y}(\mathbf{y})}\biggl[ \log \bigl( D \bigl( \mathbf{x}, \mathbf{y} \bigr) \bigr) \biggr] + \log(4)$.

---

## How to run the code

All the MI estimators are implemented in `utils.py`.

The neural networks, the class handling training/test of the estimators, and the main functions needed to run the code are implemented in `classes.py`.

The file `main.py` runs all the experiments. 
There are two running modalities that are accepted by the argument parser:
- "staircase": runs the training where the target mutual information has a staircase shape;
- "consistency_test": runs the self-consistency tests over the MNIST dataset.

Thus, you can run `main.py` by setting the argument "mode":
> python main.py --mode staircase

or by fixing the default mode of the parser to the desired one:
```default='staircase'```

---

## Citation

If you use your code for your research, please cite our paper:


---

## Contact

[nicola.novello@aau.at](nicola.novello@aau.at)

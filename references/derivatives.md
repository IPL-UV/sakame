# Derivatives of GPs

### Resources

* Differntiating GPs - [Doc](http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf)
* Gaussian Process Training with Input Noise - [Paper](http://mlg.eng.cam.ac.uk/pub/pdf/MchRas11.pdf) | [Author Website](http://mlg.eng.cam.ac.uk/?portfolio=andrew-mchutchon) | [Code](https://github.com/jejjohnson/fumadas/tree/master/NIGP)
    * Differentiating Gaussian Process - [Notes](http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf)
* Learning a Gaussian Process Model with Uncertain Inputs - [Technical Report](http://www.dcs.gla.ac.uk/~rod/publications/GirMur03-tr-144.pdf)
* Gaussian Processes: Prediction at a Noisy Input and Application to Iterative Multiple-Step Ahead Forecasting of Time-Series - [Paper](http://www.dcs.gla.ac.uk/~rod/publications/GirMur05.pdf)
* GP Regression with Noise Inputs - [Presentation](http://dcervone.com/slides/GP_noisy_inputs.pdf)
* Approximate Methods for Propagation of Uncertainty with GP Models - [Thesis](http://www.dcs.gla.ac.uk/~rod/publications/Gir04.pdf) | [Code](https://github.com/maka89/noisy-gp)
* Learning Gaussian Process Models from Uncertain Data - [Paper](https://www.researchgate.net/publication/221140644_Learning_Gaussian_Process_Models_from_Uncertain_Data) | [Github](https://github.com/maka89/noisy-gp)

---
### Covariance Functions


#### RBF Function

The typical squared exponential kernel:

$$K(x, y) = exp\left( -\frac{||x-y||^2_2}{2\lambda_d^2} \right)$$

Remember the distance calculation:

$$
d_{ij} = ||x-y||^2_2 = (x-y)^{\top}(x-y) = x^{\top}x-2x^{\top}y-y^{\top}y
$$

$$D=$$

Alternatively, one could write the kernel function in standard matrix notation:

$$k(x,y)=exp\left[-\frac{1}{2}(x-y)^{\top}\Lambda^{-1}(x-y)\right]$$

where $\Lambda$ is an ($D \times D$) matrix whos diagonal entries are $\lambda^2$.

We can also get the analytical solutions to the gradient of this kernel matrix. Which is useful for later:

$$\frac{\partial K(x,y)}{\partial \Lambda}=(x-y)^{\top}\Lambda_2^{-1}(x-y)\cdot K(x,y)$$

where $\Lambda_2$ is an ($D \times D$) matrix whos diagonal entries are $\lambda^3$.


* Gradient of RBF Kernel - [stackoverflow](https://math.stackexchange.com/questions/1030534/gradients-of-marginal-likelihood-of-gaussian-process-with-squared-exponential-co/1072701#1072701)
* Euclidean Distance Matrices (Essential Theory, Algorithms and Applications) - [Arxiv](https://arxiv.org/pdf/1502.07541.pdf)
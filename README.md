Deep Learning Exploration
=========================

This repository contains my initial exploration into recent deep learning architectures for unsupervised learning i.e. GAN, VAE, etc.  There will be more to come as I continue to bring these techniques to wireless communications, which is my area of expertise.

## Dependencies 

I made a concise effort to use as few tools as is necessary for research.  All code are made to run with Python 3.5+ and tensorflow 1.10 and their dependencies.  One extra library I use for comm. related features is CommPy

```
pip3 install git+https://github.com/veeresht/CommPy.git
```

## Implementation Notes

### Reference test scenario and architecture

I have decided to test algorithms with the mixture of Gaussian inputs as this is simple to generate and it is a reasonably hard case for unsupervised learning.  The two pattern I use repeatedly is the 8-symbol constellation, uniformed separate points on the unit circle (8-PSK in comm. speak) and the 16-QAM constellation.

For simplicity, the reference architecture for both generator and discriminator in GANs, encoder and decoder in BiGANs or VAEs will be a three hidden layer MLP with tanh only activation function only.

### Spectral normalization with GAN (SN-GAN)

We need to modify the Lipschitz norm to K=3.  This is natural consequence of tanh tending to a linear function if the support of the input is restricted to a small neighborhood about the origin, which is what happens when the largest singular value is limited to 1.  The discriminator becomes linear when the spectral norm is too small. 

# Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling
 
## Summary

<img width="587" alt="Screenshot 2021-08-09 at 12 23 19 PM" src="https://user-images.githubusercontent.com/80670240/128669623-c8961528-fff6-4d09-b4c5-dae0e6f9e8db.png">

### Encoder
The encoder is the part where we convert the image to a 200 D vector which is our latent space. 
To do this we use 5 convolution layers with BN and ReLU in between

<img width="551" alt="Screenshot 2021-08-09 at 12 23 35 PM" src="https://user-images.githubusercontent.com/80670240/128669645-5ea1d451-443e-4ad0-8b0a-5c38b88beb10.png">


### Generator
The generator uses the latent space vector to generate the 3D model. 
It contains 5 ConvTranspose3D layers with BN and ReLU in between the layers

### Discriminator
<img width="690" alt="Screenshot 2021-08-09 at 12 23 49 PM" src="https://user-images.githubusercontent.com/80670240/128669680-d581d82a-f023-4be4-a1b5-14d7e1beb291.png">

The discriminator takes the 3D Volume as input and predicts whether the 3D object is real or fake(generated model). 
The architecture of the Discriminator is the mirror of the generator model except there is a sigmoid layer attached at the end

### Loss Function
<img width="636" alt="Screenshot 2021-08-09 at 12 24 02 PM" src="https://user-images.githubusercontent.com/80670240/128669712-b27b419b-e3d7-422a-abe1-ed639164f48c.png">

x is a 3D shape from the training set, y is its corresponding 2D image, and q(z|y) is the variational distribution of the latent representation z
The loss function consists of three parts: an object reconstruction loss LRecon, a cross-entropy loss L3D-GAN for 3D-GAN, and a KL divergence loss LKL to restrict the distribution of the output of the encoder
The Kullback-Leibler Divergence score, or KL divergence score, quantifies how much one probability distribution differs from another probability distribution
So we use the KL Divergence score so that we can bring the q(z|y) as close to p(z). Basically, we want q(z|y) to represent a Gaussian distribution.

### Dataset
  * IKEA Dataset
  * SUN Database

**Pytorch Code** : https://github.com/rimchang/3DGAN-Pytorch

**Paper** : https://arxiv.org/pdf/1610.07584.pdf

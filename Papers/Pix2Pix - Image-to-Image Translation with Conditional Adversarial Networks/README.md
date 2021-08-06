# [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)

## Summary 

General-purpose solution to image-to-image translation problems.


### Objective

The generator G is trained to produce outputs that cannot be distinguished from “real” images by an adversarially trained discriminator, D, which is trained to do as well as possible at detecting the generator’s “fakes”. 

For pix2pix the objective is not just to minimise the GAN loss, i.e., ![Loss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BcGAN%7D%5Cleft%20%28%20G%2CD%20%5Cright%20%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%2Cy%7D%5Cleft%20%5B%20%5Clog%20D%5Cleft%20%28%20x%2Cy%20%5Cright%20%29%20%5Cright%20%5D%20&plus;%20%5Cmathbb%7BE%7D_%7Bx%2Cz%7D%20%5Cleft%20%5B%20%5Clog%20%5Cleft%20%28%201%20-%20D%5Cleft%20%28%20x%2C%20G%5Cleft%20%28%20x%2C%20z%20%5Cright%20%29%20%5Cright%20%29%20%5Cright%20%29%20%5Cright%20%5D) but also to minimise the ![Loss](https://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BL_1%7D%5Cleft%20%28%20G%20%5Cright%20%29%20%3D%20%5Cmathbb%7BE%7D_%7Bx%2Cy%2Cz%7D%5Cleft%20%5B%20%5Cleft%20%5C%7C%20y-G%5Cleft%20%28%20x%2Cy%20%5Cright%20%29%20%5Cright%20%5C%7C_%7B1%7D%20%5Cright%20%5D).

## Architecture

![Layout](assets/Architecture.jpg)

### Generator

Many solutions to problems in this area have used an encoder-decoder network. In such a network, the input is passed through a series of layers that progressively downsample, until a bottleneck layer, at which point the process is reversed.
To give the generator a means to circumvent the bottleneck for information like this, skip connections are added, following the general shape of a **U-Net**.

### Discriminator

In generic GANs, the discriminator cannot observe the input vector. In case of Conditional GANs, the Discriminator observes both the input as well as the Generated Vector.

Discriminator must is motivated to model high-frequency structures and L1 is relied for low-frequency structures. The discriminator tries to classify if each nxn patch. Ideal found is 70x70.

## Results

![Results](assets/Results.jpg)


## Implementation

* [Original Implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

* [Reference](https://github.com/mrzhu-cool/pix2pix-pytorch)
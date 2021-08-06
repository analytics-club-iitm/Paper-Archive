# StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks

### Pytorch Implementation of StackGAN : [StackGAN](https://github.com/Vinayak-VG/My-Projects/tree/main/Computer_Vision_Projects/Generative_Adversarial_Networks-GAN/StackGAN/scripts)

## Summary 

### Introduction

Generating Realistic Images using just text descriptions is a challenging task. There has been a lot of research under this domain, but the model does not accurately represent the text description since it is unable to capture the important features from the text descriptions

StackGAN solves the problem by generating realistic 256x256 images from just text descriptions. They divide the problem into 2 steps. Stage-I GAN outputs a 64x64 low-resolution image. The Stage-II  GAN with the 64x64 image along with the text descriptions as input outputs a 256x256 high-resolution image.

They introduce a novel Conditioning Augmentation technique that encourages smoothness in the latent space and increases the randomness in the dataset so that the model is Robust

### Previous Work
* Generative Adversarial Text to Image Synthesis: [Paper](https://arxiv.org/pdf/1605.05396.pdf)
* Learning What and Where to Draw: [Paper](https://arxiv.org/pdf/1610.02454.pdf)

### StackGAN Model

<img width="1306" alt="Screenshot 2021-07-09 at 9 25 30 AM" src="https://user-images.githubusercontent.com/80670240/125048357-74ec0e00-e0bd-11eb-8820-e13e1fa8d92e.png">

They have divided the problem into 2 modules:
* Stage-I GAN
* Stage-II GAN

We need to embed the text descriptions into an embedding vector so that they can be fed into the generator and discriminator. We can use Word2Vec or Glove for this process. But the authors recommend using the encoder used in this [paper](https://arxiv.org/pdf/1605.05395.pdf)

It is important that we retain the features from the text descriptions, otherwise we will miss those features while generating the image

### Conditioning Augmentation

If we have fewer Text-Image pairs then there is a high chance that GAN models can collapse. To prevent this, the authors use Conditioning Augmentation which encourages robustness to small disturbances

We use 2 fully connected layers, with the input as the text embedding and the output as the vector of the mean(μ0) and variance(σ0). Now we find the text conditioning variable cˆ0 by   c^0 = μ0 + σ0*ε, where ε is sampled from a normal distribution of mean = 0 and variance = 1. 

### Stage-I GAN

This part sketches the main colours and draws a rough shape of the object using the text descriptions. The input for this part is a vector where c^0 and a vector z is concatenated where z is sampled from a normal distribution of mean = 0 and variance = 1. The random noise z is used to create a random background/scenary. The Stage-I GAN generator is made of upsampling layers to generate the 64x64 low-resolution image.

The discriminator takes the 64x64x3 image as input is downsampled to 4x4x512 volume. They also compress the 1024D text embedding into a 128D vector using a fully connected layer and then spatially replicate to 4x4x128 volume. These two volumes are concatenated along the channel direction and we apply a 1x1 convolution to learn the features across the image and text. A fully connected layer with one node at the end is used to predict whether the image is real or fake

### Stage-II GAN

This part is mainly responsible to correct the defects predicted by Stage-I and capturing the important features from the text description which the Stage-I GAN left out

The output image of Stage-I GAN is used as the input along with the text embedding which fed into the Conditioning Augmentation to get the text conditioning variable c^. The Conditioning Augmentation network is not the same so that the Stage-II GAN can learn the important features left out by Stage-I GAN

The 64x64x3 image is downsampled to a 16x16x512 volume which is concatenated with spatial replication of the conditioning variable c^ to 16x16x128. This tensor(concatenated volume) is passed through a set of residual blocks that learns the representation of image and text together and it tries to learn the features from the text which the Stage-I GAN left. The output of the residual network is fed into a set of upsampling layers which outputs a 256x256 realistic image

The discriminator is modelled the same way as in the discriminator in Stage-II GAN except there are more downsampling layers since we are downsampling a 256x256x3 Image to a 4x4x512 volume in this case

### Training

During training, the discriminator takes in a real image and its corresponding text embedding as a positive sample and for negative samples, there can be two ways: First is taking a real image and a mismatched text embedding and the second is taking a synthetic image from the generator and it’s corresponding text embeddings

The generator uses the ReLU activation function whereas Discriminator uses the LeakyReLU activation function. We use LeakyReLU for smoother gradient flow through the architecture.

First, the Stage-I GAN is trained keeping Stage-II GAN fixed and  then the Stage-II GAN is trained keeping Stage-I GAN fixed

### Dataset

* CUB (Bird Species Dataset)
* Oxford-102 ( Flowers Dataset)
* MS - COCO

### Evaluation Metric (Inception Score)

<img width="347" alt="Screenshot 2021-07-09 at 1 34 09 PM" src="https://user-images.githubusercontent.com/80670240/125048766-e4fa9400-e0bd-11eb-8187-c1ba2dda76a2.png">

x denotes one generated sample, and y is the label predicted by the Inception model. The intuition behind this metric is that good models should generate diverse but meaningful images. Therefore, the KL divergence between the marginal distribution p(y) and the conditional distribution p(y|x) should be large

### Author's Results

<img width="486" alt="Screenshot 2021-07-09 at 1 46 22 PM" src="https://user-images.githubusercontent.com/80670240/125049069-31de6a80-e0be-11eb-85a7-849ee3657d75.png">

<img width="1064" alt="Screenshot 2021-07-09 at 1 46 56 PM" src="https://user-images.githubusercontent.com/80670240/125049075-33a82e00-e0be-11eb-8675-0f6323357a4f.png">

### My Results

<img width="633" alt="Screenshot 2021-08-02 at 8 32 47 PM" src="https://user-images.githubusercontent.com/80670240/127882665-ecbc9c89-f3a3-4875-845b-d652b327e020.png">
Definitely my results are not comparable with the Author's result but yeah something is better than nothing :)

### End Note

To know more about StackGAN, check out the paper: [StackGAN](https://arxiv.org/pdf/1612.03242.pdf)

To check out the Pytorch Implementation of the StackGAN model, check out my GitHub Repo: [GitHub Repo](https://github.com/Vinayak-VG/My-Projects/tree/main/Computer_Vision_Projects/Generative_Adversarial_Networks-GAN/DCGAN)

---

[Vinayak Gupta](https://github.com/Vinayak-VG)
9th July 2021

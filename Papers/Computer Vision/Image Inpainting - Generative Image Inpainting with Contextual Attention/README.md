# (Generative Image Inpainting with Contextual Attention)[https://arxiv.org/pdf/1801.07892.pdf]

## Summary 

Recent Deep Learning based methods can generate visually plausible image structures and textures but often create distorted structures or blurry textures inconsistent with the surrounding areas.

Image Inpainting or completion is the process of filling missing pixels of an image. The core challenge of image impainting lies in synthesizing visually realistic and semantically plausible pixels for the missing regions.

Image inpainting is formulated as a conditional image generation problem where high-level recognition and low-level pixel synthesis are formulated into a convolutional encoded-decoded network jointly trained with adversarial networks. 


### Approach

A unified feed-forward generative network with a novel contextual attention layer for image impainting. The network consists of two stages.

* The first stage is a simple dilated convolutional network trained with reconstruction loss to rough out the missing contents.

* The second stage is to integrate contextual attention to use the features of known patches as convolutional filters to process the generated patches. The contextual attention module also has spatial propogation layer to encourage spatial coherency. In orger to allow the network to hallucinate novel contents, another convolutional pathway is constructed in parallel with the attention pathway. The two pathways are aggregated and fed into a single decoded to obtain the final output.

The whole network is trained end to end with reconstruction losses and two GAN losses.

## Architecture

### Improved Generative Inpainting Network

A baseline generative image inpainting network by reproducing and improving state-of-the-art inpainting models.

![Layout](assets/Coarse2Fine.png)

#### Coarse-To-Fine Network

The generator network takes an image with white pixels filled in the holes and a binary mask indicating the hole regions as input pairs and outputs the final completec image. The inouut is paired with corresponding binary mask to handle holes with variable sizes, shapes and locations.

The input to the network is a 256x256 image with rectangle missing region sampled randomly during training.

In image impainting tasks, the size of receptive fields should be sufficiently large. To further enlarge the receptive fields and stabilize training, two stage coarse-to-fine network architecture is used, where,

* The first network makes an initial coarse prediction,
* The second network takes the coarse prediction as input and predicts the refined results. 

The coarse network is trained with reconstruction as well as GAN losses.

#### Global and Local Wasserstein GANs

A modified version of WGAN-GP where the GAN loss is attached to both global and local outputs of the second-stage refinement network.

For image impainting, the gradient penalty should be applied only to pixels inside the holes. This is achieved by using a mask **m**.

![Loss](https://latex.codecogs.com/gif.latex?%5Ctextup%7BLoss%7D%20%3D%20%5Clambda%20E_%7B%5Chat%7Bx%7D%5Csim%20%5Cmathbb%7BP%7D_%7B%5Chat%7Bx%7D%7D%7D%5Cleft%20%28%20%5Cleft%20%5C%7C%20%5Cnabla_%7B%5Chat%7Bx%7D%7DD%5Cleft%20%28%20%5Chat%7Bx%7D%20%5Cright%20%29%20%5Codot%20%5Cleft%20%281-%5Ctextbf%7Bm%7D%20%5Cright%20%29%29%20%5Cright%20%5C%7C_2%20-1%5Cright%20%29%5E2), where the gradient ![gradient](https://latex.codecogs.com/gif.latex?%5Cnabla_%7B%5Chat%7Bx%7D%7DD%5Cleft%20%28%20%5Chat%7Bx%7D%20%5Cright%20%29%20%3D%20%5Cfrac%7B%5Ctilde%7Bx%7D-%5Chat%7Bx%7D%7D%7B%5Cleft%20%5C%7C%20%5Ctilde%7Bx%7D-%5Chat%7Bx%7D%20%5Cright%20%5C%7C%7D) and ![xhat](https://latex.codecogs.com/gif.latex?%5Chat%7Bx%7D%20%3D%20%281-t%29x&plus;t%5Chat%7Bx%7D) where ![x](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bx%7D%20%3D%20G%28z%29) and ![z](https://latex.codecogs.com/gif.latex?z) is the input to the generator.

Intuitively, the pixel-wise reconstruction loss directly regresses holes to the current ground truth image, while WGANs implicitly learn to math potentially correct images and train the generator with adversarial gradients.

#### Spacially Discounted Reconstruction Loss

Intuitively, missing pixels near the hole boundaries have much less ambiguity than those pixels closer to the center of the hole. A spatially discounted reconstruction loss is implemented using a weight mask **M** . The weight of each pixel in the mask is computed as ![weight](https://latex.codecogs.com/gif.latex?%5Cgamma%5El) where l is the distance of the pizel to the nearest known pixel and ![gamma](https://latex.codecogs.com/gif.latex?%5Cgamma) is set to 0.99 in all experiments.

The discounted loss is more effective for improving the visual quality for larger holes.

### Image Inpainting with Contextual Attention

CNNs process image features with local convolutional kernel layer by layer are not effective for borrowing features from distant spatial locations. 

To overcome this, a attention mechanism is introduced in the deep generative layer.

![Layout](assets/ContextualAttention.png)

#### Contextual Attention

The contextual attention layer learns where to copy feature information from known background patches to generate missing patches. 

##### Match and Attend 

The problem where the features of missing pixels are to be matched with that of the surroundings (background). 

Patches are extracted in the background and then reshaped as convolutional filters. To match the foreground patches ![foreground](https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20f_%7Bx%2Cy%7D%20%5Cright%20%5C%7D) with the background ones ![background](https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20b_%7Bx%5E%5Cprime%2Cy%5E%5Cprime%7D%20%5Cright%20%5C%7D), the normalized inner product ![product](https://latex.codecogs.com/gif.latex?s_%7Bx%2Cy%2Cx%5E%5Cprime%2Cy%5E%5Cprime%7D%3D%5Cleft%20%5Clangle%20%5Cfrac%7Bf_%7Bx%2Cy%7D%7D%7B%5Cleft%20%5C%7C%20f_%7Bx%2Cy%7D%20%5Cright%20%5C%7C%7D%20%2C%20%5Cfrac%7Bb_%7Bx%5E%5Cprime%2Cy%5E%5Cprime%7D%7D%7B%5Cleft%20%5C%7C%20b_%7Bx%5E%5Cprime%2Cy%5E%5Cprime%7D%20%5Cright%20%5C%7C%7D%20%5Cright%20%5Crangle) is calculated where ![similarity](https://latex.codecogs.com/gif.latex?s_%7Bx%2Cy%2Cx%5E%5Cprime%2Cy%5E%5Cprime%7D) represents the similarity of the patch centered in the background and the foreground. Then the similarity is weighed using a scalled softmax to get attention score for each pixel ![similarity](https://latex.codecogs.com/gif.latex?s%5E*_%7Bx%2Cy%2Cx%5E%5Cprime%2Cy%5E%5Cprime%7D%20%3D%20%5Ctextup%7Bsoftmax%7D_%7Bx%5E%5Cprime%2Cy%5E%5Cprime%7D%5Cleft%20%28%20%5Clambda%20s_%7Bx%2Cy%2Cx%5E%5Cprime%2Cy%5E%5Cprime%7D%20%5Cright%20%29). The extracted patches ![background](https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20b_%7Bx%5E%5Cprime%2Cy%5E%5Cprime%7D%20%5Cright%20%5C%7D) are reused as de-convolutional filers to reconstruct foreground.

##### Attention Propogation

The coherency of attention (shift in foreground patch is likely to correspond to an equal shift in the background patch) is further increased by propogation.

To model and encourage coherency of attention maps, a left-right propogation is followed by a top-down propogation with a kernel of size k to get the new score.

![score](https://latex.codecogs.com/gif.latex?%5Chat%7Bs%7D_%7Bx%2Cy%2Cx%5E%5Cprime%2Cy%5E%5Cprime%7D%20%3D%20%5Csum_%7Bi%20%5Cin%20%5Cleft%20%5C%7B%20-k%2C%20%5Cdots%2C%20k%20%5Cright%20%5C%7D%7D%20%5Csum_%7Bj%20%5Cin%20%5Cleft%20%5C%7B%20-k%2C%20%5Cdots%2C%20k%20%5Cright%20%5C%7D%7D%20s%5E*_%7Bx&plus;i%2Cy&plus;j%2Cx%5E%5Cprime&plus;i%2Cy%5E%5Cprime&plus;j%7D)

Attention propogation significantly improves inpainting results and enriches gradients.

#### Unified Inpainting Network

A two parallel encoder is introduced to integrate the attention module. The first encoder specifically focuses on hallucinating contents with layer-by-layer (dilated) convolution, while the second encoder tries to attend on the background features of interest. Output features from two encoders are aggregated and fed into a single decoder to obtain the final output.

#### Process

Given a raw image **x** a binary mask **m** is sampled at a random location. Input image z is corrupted from the raw image as ![z](https://latex.codecogs.com/gif.latex?z%3Dx%5Codot%20m). Inpainting network ![G](https://latex.codecogs.com/gif.latex?G) takes concatenation of **z** and **m** as input and output predicted image ![output](https://latex.codecogs.com/gif.latex?x%5E%5Cprime%20%3D%20G%28z%2Cm%29) with the same size as input. Pasting the masked region of ![xprime](https://latex.codecogs.com/gif.latex?x%5E%5Cprime) to the input image, the inpainting output ![output](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bx%7D%3Dz&plus;x%5E%5Cprime%5Codot%281-m%29).


## Results

![Results](assets/Results.jpg)


## Implementation

* [Original Implementation](https://github.com/JiahuiYu/generative_inpainting)

* [Pytorch Implementation](https://github.com/DAA233/generative-inpainting-pytorch)
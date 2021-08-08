# DINO: Emerging Properties in Self-Supervised Vision Transformers

 ## Introduction
 
This model uses self-supervised learning, which allows machines to learn from random, unlabeled examples; and Vision Transformers, 
which enable AI models to selectively focus on certain parts of their input and thus reason more effectively. This model can discover
and segment objects in an image or a video with absolutely no supervision and without being given a segmentation-targeted objective.
Training ViT with DINO algorithm, it is observed that the model automatically learns an interpretable representation and separates the
main object from the background clutter. It learns to segment objects without any human-generated annotation or any form of dedicated dense pixel-level loss.

https://user-images.githubusercontent.com/46140458/116817761-47885e80-ab68-11eb-9975-d61d5a919e13.mp4

## Summary

The core component of Vision Transformers are self-attention layers. In this model, each spatial location builds its representationby “attending” to the other 
locations. That way, by “looking” at other, potentially distant pieces of the image, the network builds a rich, high-level understanding of the scene.
When visualizing the local attention maps in the network, we see that they correspond to coherent semantic regions in the image. DINO works by interpreting self-supervision
as a special case of self-distillation, where no labels are used at all. Indeed, a student network is trained by simply matching the output of a teacher network over 
different views of the same image. These different views are generated using multi-cropping technique.
 
### Network Structure


<div align="center">
  <img width="100%" alt="DINO illustration" src="https://github.com/wigglytuff-tu/dino/blob/main/.github/dino.gif">
</div>

Self-distillation with no labels. Image(below) illustrates DINO in the case of one single pair of views (x1, x2) for simplicity. The model passes two different random transformations of an input image to the student and teacher networks. Both networks have the same architecture but different parameters. The output of the teacher network is centered with a mean computed over the batch. Each networks outputs a K dimensional feature that is normalized with a temperature softmax over the feature dimension. Their similarity is then measured with a cross-entropy loss. A stop-gradient (sg) operator is applied on the teacher network to propagate gradients only through the student. The teacher parameters are updated with an exponential moving average (ema) of the student parameters.
<div align="center">
  <img width="50%" alt="DINO illustration" src="/Papers/DINO: Emerging Properties in Self-Supervised Vision Transformers/assets/Screenshot 2021-08-08 194450.jpg">
</div>

### Result
Self-attention from a Vision Transformer with 8 × 8 patches trained with no supervision. We look at the self-attention of
the [CLS] token on the heads of the last layer. This token is not attached to any label nor supervision. These maps show that the model
automatically learns class-specific features leading to unsupervised object segmentations.

<div align="center">
  <img width="100%" alt="Self-attention from a Vision Transformer with 8x8 patches trained with DINO" src="https://github.com/wigglytuff-tu/dino/blob/main/.github/attention_maps.png">
</div>

### Datsets used:
1. ImageNet
2. DAVIS
3. DAVIS 2017
4. YFCC100M

### Links
1. [Research Paper](https://arxiv.org/abs/2104.14294v2)
2. [Code](https://github.com/facebookresearch/dino)

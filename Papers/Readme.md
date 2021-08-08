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


<div align="center">
  <img width="100%" alt="Self-attention from a Vision Transformer with 8x8 patches trained with DINO" src="https://github.com/wigglytuff-tu/dino/blob/main/.github/attention_maps.png">
</div>

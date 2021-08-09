# Pix2Vox++: Multi-scale Context-aware 3D Object 

## Summary

<img width="716" alt="Screenshot 2021-08-09 at 12 39 39 PM" src="https://user-images.githubusercontent.com/80670240/128670965-98b3e00c-923a-4818-a5b8-9dcf84bd22ae.png">
<img width="716" alt="Screenshot 2021-08-09 at 12 39 48 PM" src="https://user-images.githubusercontent.com/80670240/128670969-67e8ef0f-d76f-489c-bb18-a2489d7296fb.png">

### Encoder

The part of the encoder is made up of Resnet 18 or 50 depending on the model Pix2Vox++/F or Pix2Vox++/A
At the end of the encoder, we attach 3 Convolution layers so that the output of the encoder can be made available for the decoder.

### Decoder

The decoder part is completely made up of 3D convs so that we can get the desired 3D model from the 2D image
If we want to output high-resolution models then we just need to increase the number of convnets and also the number of channels used in the 3D convnets

### Multi-scale Context-aware Fusion

<img width="634" alt="Screenshot 2021-08-09 at 12 39 58 PM" src="https://user-images.githubusercontent.com/80670240/128670993-86702a31-4eab-4e66-a1a1-34ff8c1d79ca.png">

This part is responsible for Multi-view 3D reconstruction. Basically, it selects the high-quality portion from the coarse 3D volumes and then fuses the 3D model in such a way that only the best portion is selected.
So basically it tries to give a score to each voxel of the 3D model based on its priority and quality. Then we multiply the score map with course volume and then add all of them to get the final 3D model.
We also try to concatenate multiple feature maps so that we don’t lose the important previous features while calculating the score map.

<img width="648" alt="Screenshot 2021-08-09 at 12 40 07 PM" src="https://user-images.githubusercontent.com/80670240/128671008-fab9a806-9065-4e61-990a-e662f91c1958.png">

### Refiner

<img width="632" alt="Screenshot 2021-08-09 at 12 40 17 PM" src="https://user-images.githubusercontent.com/80670240/128671022-b09c3a41-1eaa-48ce-91e1-87bb9d729926.png">

The refiner can be seen as a residual network, which aims to correct the wrongly recovered parts of a 3D volume. It follows the concept of a 3D encoder-decoder with U-net connections that preserves the local structure in the fused volume.

### Training

We first train the model without the multi-scale fusion using only single view images and then train the network with multi-scale context-aware fusion for multiple view images.

### Dataset
  * Shapenet
  * Pix3D
  * Things3D

**Pytorch Code** : [Code](https://gitlab.com/hzxie/Pix2Vox)

**Paper** : [Paper](https://arxiv.org/pdf/2006.12250.pdf)










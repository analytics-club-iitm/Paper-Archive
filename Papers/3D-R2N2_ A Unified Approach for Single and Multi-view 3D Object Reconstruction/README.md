# 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction

## Summary

### Introduction

<img width="448" alt="Screenshot 2021-08-06 at 1 22 12 PM" src="https://user-images.githubusercontent.com/80670240/128476485-2c2dc6ea-0269-4a67-a967-48052a7b3674.png">
<img width="1045" alt="Screenshot 2021-08-06 at 1 22 33 PM" src="https://user-images.githubusercontent.com/80670240/128476535-59876b06-1c77-41c4-85b3-25db1b722872.png">

They have used LSTM, which is useful for storing information when we use Multi-view images for reconstruction. 
Basically, we feed the information one image at a time, so if we want to reconstruct using 3 orthogonal projections. After feeding the first image, now when we feed the next image, then basically the LSTM remembers the previous image and tries to reconstruct using the previous information
So in training when we feed in a new image, then the LSTM forgets the previous information and retains only the important information(generic to all images)
We make use of Residual Connections like Resnet to speed up optimization

### 3D LSTM

<img width="1009" alt="Screenshot 2021-08-06 at 1 22 48 PM" src="https://user-images.githubusercontent.com/80670240/128476563-94c1486d-bd14-4d69-9c88-c37d1e7ee5c1.png">

The Fully connected is passed on to the LSTM. This is a 3D LSTM. In a vanilla-LSTM, all elements in the hidden layer ht-1 affect the current hidden state hₜ but in 3D LSTM only the neighbouring LSTM affecٖt the hidden state.
Basically, ht-1 is 3x3x3 space since it receives the previous state from the neighbouring LSTM and we apply 3x3x3 conv to convert it to a proper dimension
Each unit in the LSTM learns to reconstruct only a part of the voxel space and not the entire space. This also gives a sense of locality so it can selectively update its prediction. 
We can also use GRU instead of LSTM and results have shown that GRU performs much better than LSTM

### Decoder

After receiving an input image sequence x1, x2, · · ·, xT, the 3D-LSTM passes the hidden state hₜ to a decoder, which increases the hidden state resolution by applying 3D convolutions, non-linearities, and 3D unpooling until it reaches the target output resolution.

### Dataset used

   * Shapenet
   * PASCAL 3D

### EndNote 

To know more about the 3D-R2N2 Model, check out the paper: [3D-R2N2](https://arxiv.org/pdf/1604.00449.pdf)

You can check out the pytorch code at [Pytorch Code](https://github.com/chrischoy/3D-R2N2)

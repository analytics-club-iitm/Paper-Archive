# UNet++: A Nested U-Net Architecture for Medical Image Segmentation

## Summary 

### Introduction

Biomedical Image Segmentation has been a centre of attention in recent years. Achieving very high accuracy is very important in the field of medicine since it decides your life. Hence to improve the accuracy of Image Segmentation, UNet++ was introduced which was a further development on the previously U-Net Model.

To check out my previous report on U-Net Paper, check out [U-Net](https://github.com/Vinayak-VG/My-Projects/tree/main/Computer%20Vision%20Projects/2D%20Image%20Segmentation/U-Net%20Image%20Segmentation/U-Net)

In the U-Net model, we concatenate the feature maps from the encoder part to the decoder part. But these two feature maps are semantically very different. Concatenating them directly will make the model difficult to learn the localisation information from the context information. Hence the authors have constructed a new architecture UNet++ which bridges the semantic gap between the encoder and decoder by connecting the 2 parts with a series of nested, dense skip pathways

### Previous Work

U-Net: Convolutional networks for biomedical image segmentation: [Link](https://arxiv.org/pdf/1505.04597.pdf)

### UNet++

<img width="634" alt="Screenshot 2021-07-03 at 11 41 58 AM" src="https://user-images.githubusercontent.com/80670240/124345028-c8f57f00-dbf3-11eb-86d6-fa7e3465a7cb.png">

Each circle represents a series of 3x3 convolution layers with Batch Norm and ReLU function. The black circles which form the V-Shaped are the backbone of the model and it is the original U-Net model. The green circles in the centre are responsible for bridging the gap between the encoder and the decoder. These green circles represent dense convolutional blocks. They are connected by skip pathways just like in DenseNet. This helps to retain the context information longer and also helps in the gradient flow

The input for the green circles comes from two sources. First from the skip connections and second from the circle diagonally below it. 
Since it comes from a layer below it, the output feature map will not match the input size of the green circle. So the output from that circle is up transposed to the required dimensions and then concatenated with the input. 

The red lines represent deep supervision. Instead of learning only from the final output, we learn also from the intermediate segmentation maps. We calculate the loss function for each intermediate segmentation maps and also from the final segmentation map and then we add all the losses. This significantly decreases the training time.

Loss Function

They use a mix of Binary Cross-Entropy and Dice Loss. 

Loss = ½BCE + DiceLoss

<img width="609" alt="Screenshot 2021-07-03 at 11 43 32 AM" src="https://user-images.githubusercontent.com/80670240/124345058-fb06e100-dbf3-11eb-9d32-8b7f8a6785bf.png">

The reason why we use Dice Loss is that it considers both local and global information when calculating the loss whereas the BCE only calculates discretely. Also, it solves the problem of class imbalance. To read more about Dice Loss, check out Dice Loss

### Accuracy Metric

They have used the IoU score instead of normal pixel-pixel comparison because IoU takes care of class imbalance. If you see the segmentation map, it has a lot of white spaces when compared to the black border. Hence if we use pixel-pixel comparison we may get a high accuracy but it could be misleading. Hence they have used the IoU score

### Training

Optimizer: Adam Optimizer

Batch Size: 1

Learning Rate: 1e-3

Loss Function: BCE + Dice Loss

### Results

UNet++ : IoU Score 97.8%
This clearly shows the superiority of UNet++ over the Vanilla U-Net. Vanilla U-Net gave an IoU score of 95.3% whereas UNet++ is giving a score of 97.8%
This shows that the bridging of the semantic gap between the encoder and decoder helped to get more accurate results

<img width="637" alt="Screenshot 2021-07-03 at 11 43 46 AM" src="https://user-images.githubusercontent.com/80670240/124345063-00642b80-dbf4-11eb-96ac-e4f9acec31d7.png">

### End Note

To know more about U-Net++, check out the paper: [U-Net++](https://arxiv.org/pdf/1807.10165.pdf)

To check out the Pytorch Implementation of the U-Net model, check out my GitHub Repo: [GitHub Repo](https://github.com/Vinayak-VG/My-Projects/tree/main/Computer_Vision_Projects/2D_Image_Segmentation/U-Net_Image_Segmentation/U-Net%2B%2B/scripts)

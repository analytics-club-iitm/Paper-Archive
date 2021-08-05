# U-Net: Convolutional Networks for Biomedical Image Segmentation

## Summary

### Introduction

Image Segmentation is evolving very fast in the field of Machine Learning and Computer Vision. In this era, we expect ML to solve many real-life problems with high accuracy. But this is not in the field of medicine. In the field of medicine, we need a very high level of accuracy to consider a problem to be solved and unfortunately, this is not the case in Biomedical Image Segmentation. 

Hence the authors have proposed the U-Net Architecture which is not only accurate but also achieves very high accuracy only with very little training data. The authors have proposed a new data augmentation technique to handle the small training dataset. This architecture won the ISBI Challenge for Segmentation of neuronal structures in EM Stacks

### Previous Work

Deep Neural Networks Segment Neuronal Membranes in Electron Microscopy Images: [Link](https://proceedings.neurips.cc/paper/2012/file/459a4ddcb586f24efd9395aa7662bc7c-Paper.pdf)

Image Segmentation with Cascaded Hierarchical Models and Logistic Disjunctive Normal Networks: [Link](https://openaccess.thecvf.com/content_iccv_2013/papers/Seyedhosseini_Image_Segmentation_with_2013_ICCV_paper.pdf)

### U-Net Model

<img width="789" alt="Screenshot 2021-07-03 at 11 34 14 AM" src="https://user-images.githubusercontent.com/80670240/124344870-b7f83e00-dbf2-11eb-85f8-e359982cda03.png">

The U-Net Model is made up of 2 parts, Encoder and Decoder. The Encoder(first half of the network) consists of convolution layers and max pool layers. The encoder part is responsible for capturing the context/information from the image. The Decoder(second half of the network) consists of convolution layers and Up Transpose Convolution layers. The decoder captures the localisation. 

This can be visualised as follows: Take for example a cat in an image. So basically the encoder part learns to recognise that there is a cat in the picture. But in the process of capturing the context of the image, it has lost the localisation information i.e where exactly was the cat in the image? To recover the localisation information, the decoder increases the resolution of the output in each layer by the Up Convolution layers. This way the U-Net Architecture is able to capture the information from the image accurately

To help the decoder better capture localisation, the final output from each layer of the encoder is concatenated with the output after every Up Convolution in the decoder layer. This helps to better capture the localisation information and helps in faster training

### Data Augmentation

In the ISBI Cell Segmentation challenge, we were given only 30 electron microscopy images and their corresponding segmentation maps. This is too few images to get good accuracy. Hence the authors used random elastic deformation to the training data which gives robustness and randomness to the training data and hence reduces the chance of overfitting to the training set

<img width="627" alt="Screenshot 2021-07-03 at 11 34 47 AM" src="https://user-images.githubusercontent.com/80670240/124344874-c0507900-dbf2-11eb-827e-4fa575a3bdfd.png">

The left side image is the real image and the right side image is the deformed image of the left side

### Training

Optimizer: Stochastic Gradient descent

Batch size: 1

Loss Function: Pixel-wise Binary cross-entropy

### Results

<img width="637" alt="Screenshot 2021-07-03 at 11 34 35 AM" src="https://user-images.githubusercontent.com/80670240/124344876-c47c9680-dbf2-11eb-8fc2-2cf786466e41.png">
IoU Score: 95.3%

### End Note
To know more about the U-Net Model, check out the paper: [U-Net](https://arxiv.org/pdf/1505.04597.pdf)

To check out the Pytorch Implementation of the U-Net model, check out my GitHub Repo: [GitHub Repo](https://github.com/Vinayak-VG/My-Projects/tree/main/Computer_Vision_Projects/2D_Image_Segmentation/U-Net_Image_Segmentation/U-Net/scripts)

PS: I have included the train and test data in the GitHub repository. Unfortunately, the segmentation maps for the test data are unavailable.  

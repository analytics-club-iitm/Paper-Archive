# PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

### Pytorch Implementation of PointNet for Part Segmentation : [PointNet](https://github.com/Vinayak-VG/My-Projects/tree/main/Computer_Vision_Projects/3D_Segmentation/3D_Part_Segmentation/PointNet/scripts)

## Summary

### Introduction

Usually, Point Cloud representations are converted to mesh or voxel representations because of their irregularity. But converting them into mesh or voxels causes unnecessary addition of data and causes issues. This paper presents a novel architecture that takes in the point cloud directly. This model allows for object classification, part segmentation, to scene semantic parsing.

<img width="579" alt="Screenshot 2021-07-09 at 5 40 39 PM" src="https://user-images.githubusercontent.com/80670240/125079244-028d2500-e0e1-11eb-8b09-28e4f1a93ff9.png">

### Invariance of the model

We need to make sure that the model is not affected by the variance of the point cloud. Since in a point cloud, the points are spread irregularly it is not possible to take the points in order. Hence, if the points are taken in any order for input, the output prediction should always be the same no matter the order of points taken. Also, the model should take care of noise/missing data in the point cloud. 

Suppose a point in the cloud is disturbed by a small amount, this disturbance should not affect the model predictions since it is common in Point Cloud representation for small noises. Since the points are not isolated and are always closely spaced in the point cloud, we should also consider the local features/information wrt to that point while making predictions.


### PointNet Architecture

<img width="845" alt="Screenshot 2021-07-03 at 11 21 25 AM" src="https://user-images.githubusercontent.com/80670240/124344602-d2c9b300-dbf0-11eb-9dfb-e294a7e5198d.png">

<img width="717" alt="Screenshot 2021-07-09 at 5 42 25 PM" src="https://user-images.githubusercontent.com/80670240/125079352-1fc1f380-e0e1-11eb-857c-33dc7a11335f.png">

The above network is used exclusively for Part-Segmentation

The PointNet is a simple but efficient and powerful model. We will concentrate on the segmentation network more. The PointNet architecture is made up of MLPs, transformation networks and max-pooling. For Part Segmentation, we need to have both global and local features since we draw the boundary locally. Hence we use both global and local features in our network for the Segmentation task

To make the model invariant to the input data, we create a mini-network that predicts a 3x3 transformation grid, which is multiplied with the input matrix to get an invariant matrix. Basically think in this way: Assume the point cloud in the 3D space is rotated by an angle θ. So basically this 3x3 grid is kind of a rotational matrix. So using the input data, the mini-network predicts the appropriate 3x3 grid which rotates the point cloud in the opposite direction(rotates the point cloud by an angle -θ). So now given any rotation of the input point cloud, the model removes the rotation variance of the input point cloud using this mini neural network

<img width="693" alt="Screenshot 2021-07-02 at 11 32 27 PM" src="https://user-images.githubusercontent.com/80670240/124344611-e1b06580-dbf0-11eb-8c0c-43a282f39123.png">
The above picture is the architecture of T-Net model

Similarly, we predict extend this method to remove the variance for the features as well by predicting a feature transformation using a similar mini-network but this time we are using a larger grid size. So to avoid overfitting, we add a regularization term. A is the feature transformation matrix

<img width="283" alt="Screenshot 2021-07-02 at 11 40 00 PM" src="https://user-images.githubusercontent.com/80670240/124344618-fa208000-dbf0-11eb-8cb1-8a6b2c375ceb.png">

The point cloud representation is an unordered representation when compared to pixels in a 2d image or voxels in a 3d volume. Hence sampling points in any order should not affect the model performance and the model should always almost predict the same output

<img width="205" alt="Screenshot 2021-07-09 at 5 46 43 PM" src="https://user-images.githubusercontent.com/80670240/125079655-834c2100-e0e1-11eb-9bb2-50623a5f147a.png">

So to tackle this problem, we use the Max-Pooling layer which basically removes the problem of unorderedness. Since we do max pooling across the layers, it really does not matter the order we sample the points from the point cloud
We concatenate the local and global features which makes optimisation faster since for Part Segmentation, it is important for the model to be aware of the local variables as well

### Dataset 

We evaluate on [ShapeNet Part Dataset](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip), which contains 16,881 shapes from 16 categories, annotated with 50 parts in total. Most object categories are labelled with two to five parts. Ground truth annotations are labelled on sampled points on the shapes. We formulate part segmentation as a per-point classification problem.

### Evaluation Metric

The evaluation metric is mIoU on points. For each shape S of category C, to calculate the shape’s mIoU: For each part type in category C, compute IoU between ground truth and prediction. If the union of ground truth and prediction points is empty, then count part IoU as 1. Then we average IoUs for all part types in category C to get mIoU for that shape. To calculate mIoU for the category, we take the average of mIoUs for all shapes in that category.

### Training
No Dropouts used. The [momentum for batch normalization](https://medium.com/@ilango100/batchnorm-fine-tune-your-booster-bef9f9493e22) starts with 0.5 and is gradually increased to 0.99. We use adam optimizer with an initial learning rate 0.001, momentum 0.9 and batch size 32. The learning rate is divided by 2 every 20 epochs.

Loss Function: Softmax classification loss + Regularisation loss

Accuracy Metric: IoU Score and Per-Point classification accuracy

### Results
![BeFunky-collage](https://user-images.githubusercontent.com/80670240/127381364-cc8515b4-ff7b-499e-83f0-a157d344e376.jpg)
Accuracy : 91.9%

### EndNote
To get to know more about the model, check out the paper: [PointNet](https://arxiv.org/pdf/1612.00593.pdf)

---

[Vinayak Gupta](https://github.com/Vinayak-VG)
3rd July 2021





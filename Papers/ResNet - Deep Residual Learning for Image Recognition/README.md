# [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

## Summary

As the number of layers in a deep neural network increases, training becomes increasingly more difficult. <br> This leads to issues such as *vanishing gradients*, *accuracy degradation*, and a *longer training time*.
<br>
Eventually, it led to the development of the ResNet, which allows deeper models having less susceptibility to these issues.

## Maths Involved
![Residual block maths](https://github.com/dj-dg/Paper-Archive/blob/master/Papers/ResNet%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition/assets/Screen%20Shot%202021-12-23%20at%202.46.31%20PM.png "Residual block maths")
<br> 
Here, **x** and **y** are the input and output vectors of the given layer. <br>
**F(x,{Wi})** represents the residual mapping to be learned. <br>
Here, the dimensions of **F** and **x** should be the same so that they can be added; else, according conversions need to be done.
<br>
![Residual block structure](https://github.com/dj-dg/Paper-Archive/blob/master/Papers/ResNet%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition/assets/Screen%20Shot%202021-12-23%20at%203.23.24%20PM.png "Residual block structure")
<br>

## Architecture Examples, and Results

The ResNet is not an architecture as such, but rather implements the residual module as shown above.
<br>
However, the authors of the paper did implement a few models using residual blocks and published their results for the same, evaluated on the ImageNet 2012 dataset that consists of 1000 classes.
<br><br>
The following image shows how one can implement residual blocks in a normal network, to retain the layers while increasing efficiency. <br>
![Normal network to Resnet](https://github.com/dj-dg/Paper-Archive/blob/master/Papers/ResNet%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition/assets/Screen%20Shot%202021-12-23%20at%203.23.40%20PM.png "Normal to ResNet")
<br><br>
Here are some architectures using ResNets. <br>
![Other Architectures](https://github.com/dj-dg/Paper-Archive/blob/master/Papers/ResNet%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition/assets/Screen%20Shot%202021-12-23%20at%203.23.59%20PM.png "Other architectures")
<br>
They trained models of different numbers of layers (18, 34, 50, 101, 152 layers) and published the results in the original paper.
This is a training curve of normal versus residual networks of same number of layers. <br> <br>
![Training Graph](https://github.com/dj-dg/Paper-Archive/blob/master/Papers/ResNet%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition/assets/Screen%20Shot%202021-12-23%20at%203.23.53%20PM.png "Training Graph")
<br>

Here is a comparison of the results obtained in the paper, from implementing different models. <br><br>
![Results](https://github.com/dj-dg/Paper-Archive/blob/master/Papers/ResNet%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition/assets/Screen%20Shot%202021-12-23%20at%203.24.27%20PM.png "Results")
<br>




## Further Reading
You can find the original paper [here](https://arxiv.org/pdf/1512.03385.pdf).
<br>
Here is the link to a repo containing the [PyTorch implementation](https://github.com/a-martyn/resnet).
<br>
Here is a link to a simple ResNet model I have implemented myself: [Basic Image Classifier](https://github.com/dj-dg/image-classifier-basic)

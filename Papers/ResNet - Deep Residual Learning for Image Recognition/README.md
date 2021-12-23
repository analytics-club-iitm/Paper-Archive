# [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

## Summary

As the number of layers in a deep neural network increases, training becomes increasingly more difficult. <br> This leads to issues such as *vanishing gradients*, *accuracy degradation*, and a *longer training time*.
<br>
Eventually, it led to the development of the ResNet, which allows deeper models having less susceptibility to these issues.

## Maths Involved
(insert image here)
Here, **x** and **y** are the input and output vectors of the given layer. <br>
**F(x,{Wi})** represents the residual mapping to be learned. <br>
Here, the dimensions of **F** and **x** should be the same so that they can be added; else, according conversions need to be done.
<br>
(insert image of building block)

## Architecture Examples, and Results

The ResNet is not an architecture as such, but rather implements the residual module as shown above.
<br>
However, the authors of the paper did implement a few models using residual blocks and published their results for the same, evaluated on the ImageNet 2012 dataset that consists of 1000 classes.
<br>
They trained models of different numbers of layers (18, 34, 50, 101, 152 layers) and published the results in the original paper.

(insert architecture example)
(insert results of 18 and 34 layers)


## Further Reading
You can find the original paper [here](https://arxiv.org/pdf/1512.03385.pdf).
<br>
Here is the link to a repo containing the [PyTorch implementation](https://github.com/a-martyn/resnet).
<br>
Here is a link to a simple ResNet model I have implemented myself: [Basic Image Classifier](https://github.com/dj-dg/image-classifier-basic)

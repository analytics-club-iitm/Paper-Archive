# Learning Implicit Fields for Generative Shape Modeling

## Summary 

<img width="616" alt="Screenshot 2021-08-09 at 8 18 16 AM" src="https://user-images.githubusercontent.com/80670240/128655660-4117e0ba-687c-4a6d-bbf0-316d6cfa85ea.png">

### Encoder
The encoder is made up of ResNet Architecture

It basically converts the image to a 128D Feature Vector

### Decoder 

<img width="565" alt="Screenshot 2021-08-09 at 8 18 31 AM" src="https://user-images.githubusercontent.com/80670240/128655672-1c798502-54de-423a-b907-2d2281b177e5.png">

The decoder takes in 2 inputs. One is the feature vector and another is the point coordinate (x, y, z).
We sample out every point in the 3D space(64x64x64) and feed it into the network and we get a prediction of either 1 or 0 which tells whether the point coordinate lies inside or outside the 3D object.
We concatenate the fully connected layers so that we can optimise faster.
We obtain a voxel representation of the 3D object and we apply the Marching Cubes method to convert the voxels into a mesh representation of the 3D object.

### Loss Function

<img width="555" alt="Screenshot 2021-08-09 at 8 18 44 AM" src="https://user-images.githubusercontent.com/80670240/128655682-1610cbf4-bc32-4ea8-a77f-6ebbc6c12ffb.png">

f is the predicted output and F is the ground truth.
We assign high weights to the points which are in a high-density area(i.e near the 3D object) and assign low weights to the points which are far away from the 3D object	

### Dataset
  * Shapenet

### EndNote

To know more about the Model, check out the paper: [Paper](https://arxiv.org/pdf/1812.02822.pdf)

You can check out the pytorch code at [Pytorch Code](https://github.com/czq142857/implicit-decoder)

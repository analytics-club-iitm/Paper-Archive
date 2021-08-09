# Occupancy Networks: Learning 3D Reconstruction in Function Space

## Summary

### Introduction

<img width="612" alt="Screenshot 2021-08-09 at 12 30 52 PM" src="https://user-images.githubusercontent.com/80670240/128670364-baec1d7e-9a13-4c30-ac3c-dcbdaabd4465.png">

This paper is similar to the implicit network paper, but the only difference is that instead of sequencially passing the coordinates, they parallelly pass all the coordinates together to make the best use of GPU

<img width="624" alt="Screenshot 2021-08-09 at 12 31 04 PM" src="https://user-images.githubusercontent.com/80670240/128670389-dfa4acd6-84a3-455b-ac96-b263f95e4426.png">

### Multiresolution Isosurface Extraction

<img width="634" alt="Screenshot 2021-08-09 at 12 31 17 PM" src="https://user-images.githubusercontent.com/80670240/128670396-cbe5e784-ff50-4bfe-870e-561980971fb5.png">

We first mark all points at a given resolution that have already been evaluated as either occupied (red circles) or unoccupied (cyan diamonds). We mark all grid points p as occupied for which fθ(p,x) is bigger or equal to some threshold τ. 
We then determine all voxels that have both occupied and unoccupied corners and mark them as active (light red) and subdivide them into 8 subvoxels each.
Next, we evaluate all new grid points (empty circles) that have been introduced by the subdivision. The previous two steps are repeated until the desired output resolution is reached.
Finally, we extract the mesh using the marching cubes algorithm, simplify and refine the output mesh using first and second-order gradient information.
	
### Metrics
  * IoU
  * Chamfer - L1
  * Normal Consistency Score

### Dataset
  * ShapeNet
  * KITTI
  * Online Products dataset

**Pytorch Code** : [Code](https://github.com/autonomousvision/occupancy_networks)

**Paper** : [Paper](https://arxiv.org/pdf/1812.03828.pdf)



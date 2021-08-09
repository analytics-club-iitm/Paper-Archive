# AlphaFold

# Improved protein structure prediction using potentials from deep learning

## Summary

### Introduction

Protein folding is one of the holy grail problems in biology. Predicting protein structure from the chain of amino acids was one of the biggest challenges and it was always thought that AI should be able to solve it. 

They trained a neural network to predict the distance between the residues and the torsional angles from the MSA(multiple sequence alignment) features of the protein. Using the distance and the torsional angles, using just simple gradient descent they were able to achieve state-of-the-art performance.

### AlphaFold

Using the MSA features, we use deep neural networks to learn the distance matrix and the torsional angles(phi and psi). To avoid memory overload and overfitting we predict these features only using a 64x64 window like as shown in the figure below. It is also shown that contact predictions need only a small local window. 

<img width="618" alt="Screenshot 2021-08-06 at 1 45 38 PM" src="https://user-images.githubusercontent.com/80670240/128479961-a555a79e-4030-4029-976f-e6a76090c5c4.png">

To find the distance matrix for all the LxL residue pairs, we combine these individual 64x64 maps. We use 220 residual blocks with dilated convolutions to predict the features. We use ELU non-linearity and stochastic gradient descent

<img width="711" alt="Screenshot 2021-08-06 at 1 47 50 PM" src="https://user-images.githubusercontent.com/80670240/128479996-d0660b4e-963b-4e44-9cbc-c21946afa906.png">

After predicting the distance matrix and the torsional distributions, now it’s time to predict the 3d protein structure. This is done by minimizing the potential using gradient descent. They define 3 different types of potentials: Distance Potential, Torsional Potential and Vanderwaal Potential. 

The distance potential is the negative log-likelihood of the distance summed over all the residue pairs. The torsional potential is also modelled as negative log-likelihood. Since we predict the marginal distribution, we fitted a unimodal von Mises distribution to the marginal predictions. Also taking into account the steric repulsions between the amino acids, we add a vanderwaal’s term to take care of that

<img width="671" alt="Screenshot 2021-08-06 at 1 48 00 PM" src="https://user-images.githubusercontent.com/80670240/128480089-bc977e91-d4be-4783-bbd4-158aada1571a.png">

Now after defining the potentials, they initialise a 3d protein structure by sampling from the torsion distribution. They run gradient descent(L-BFGS) algorithm and they obtain the 3d protein structure

<img width="637" alt="Screenshot 2021-08-06 at 1 48 09 PM" src="https://user-images.githubusercontent.com/80670240/128480117-1fe1e7d0-a17a-47e8-8ac6-cb974bdb1f40.png">

To generalise and to increase randomness in the dataset, we use different offsets of 64x64 maps from the distogram which in turn creates new training data. Also after predicting the final 3d structure we add noise to the coordinates and run gradient descent again. This creates randomness to the training set and hence reduces overfitting

### Accuracy Methods
To score the final 3d model, they have used different metrics like  TM score, GDT(Global Distance Test) and RMS Distance. These require geometric alignment. So they used a new metric called 	IDDT	

### EndNote

To know more about AlphaFold, check out the paper by DeepMind: [AlphaFold](https://www.nature.com/articles/s41586-019-1923-7.epdf?author_access_token=Z_KaZKDqtKzbE7Wd5HtwI9RgN0jAjWel9jnR3ZoTv0MCcgAwHMgRx9mvLjNQdB2TlQQaa7l420UCtGo8vYQ39gg8lFWR9mAZtvsN_1PrccXfIbc6e-tGSgazNL_XdtQzn1PHfy21qdcxV7Pw-k3htw%3D%3D)

To know more about the implementation of AlphaFold, check out their GitHub Repository [Code](https://github.com/deepmind/deepmind-research/tree/master/alphafold_casp13)

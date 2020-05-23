# Neural_network
Multi hidden layers neural network for classification as generalization from Stanford Class CS229 on Machine learning

NN_mhl contains the code that launches the multi-hidden-layers neural network ("NN")
Inputs are required in lines 10 to 16 of the file NN_mhl and from line 12 of NN_mhlcv
NN_mhl loads a dataset xxx.mat with a matrix X of size (m,n) and a matrix y of size (m,1)

NN_mhlcv is the same code but allows for cross-validation by splitting the dataset into 2 parts :
 - the training set (whose size is defined by train_size thats is = or < than m)
 - the cross-validation set (that runs on X starting from position cv_size_init to m)
 
Several datasets with manually written numbers from 0 to 9 with black (_B), white (_W) or grey (_G) backgrounds or with combinations thereof (_BW, _BGW) are provided, including the original dataset from Stanford CS229 class (data_G.mat).

The optimisation function minimize a continuous differentialble multivariate function. 
Starting point is given by "X" (D by 1), and the function named in the string "f", must return a function value and a vector of partial derivatives. 
The Polack-Ribiere flavour of conjugate gradients is used to compute search directions, and a line search using quadratic and cubic polynomial approximations and the Wolfe-Powell stopping criteria is used together with the slope ratio method for guessing initial step sizes.

# Neural_network
Multi hidden layers neural network for classification as generalization from Stanford Class CS229 on Machine learning

NN_mhl contains the code that launches the multi-hidden-layers neural network ("NN")
Inputs are required in lines 10 to 16 of the file NN_mhl and from line 12 of NN_mhlcv
NN_mhl loads a dataset xxx.mat with a matrix X of size (m,n) and a matrix y of size (m,1)

NN_mhlcv is the same code but allows for cross-validation by splitting the dataset into 2 parts :
 - the training set (whose size is defined by train_size thats is = or < than m)
 - the cross-validation set (that runs on X starting from position test_size_init to m)
 

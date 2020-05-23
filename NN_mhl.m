%% Neural Network multi-hidden layers by Jean Dessain
%% This file is a development from the CS229 Standford class 
%% The original exercise was dedicated to single hidden layer neural network

%% =========== Part 1: Initialization
% clear ; close all; clc
fprintf('Initializing ...\n')

%% Setup the parameters for defining the neural network
data_file = 'data_BW1.mat';         % choose the dataset for training and cross-validation from the .mat list or from your own
input_layer_size  = 400;            % original - 20x20 Input Images of Digits
num_hl = 3;                         % number of hidden layers in the network + initial layer X = A1 and final layer - min = 2
hl_size = 25;                       % number of units per hidden layer
num_labels = 10;                    % 10 labels, from 1 to JDE 11 as some images are not numbers   
Max_iter = 100;                     % Number of iterations or training the neural network
lambda = 1;                         % coefficient of normalisation of the network - the higher lambda, the higher the bias and the lower the variance

%% =========== Part 2: Loading and Visualizing Data =============
%% +++ Load Training Data with two possible datasets +++

%% A. Loading dataset from .mat file
load(data_file);
%% End version A.
 
%% OR  %%
 
%% B. loading dataset from an excel file
% pkg load io;
% pkg load windows;
% XLS = xlsread('Mat_excel_BW.xlsx');
% X = XLS(2:end, 2:401);
% y = XLS(2:end, 403);
%% End Version B.
%% End loading data

m = size(X, 1);

%% +++  Visualisation of a subset of X - OPTIONAL +++
%% Randomly select 100 data points to display the type of figures analyzed
% fprintf('Visualizing Data ...\n')
% sel = randperm(size(X, 1));
% sel = sel(1:100);
% displayData(X(sel, :));
% fprintf('Program paused. Press enter to continue.\n');
% pause;


%% ================ Part 3: Initializing Pameters ================
%% Starting by implementing a function to initialize the weights of the neural network
%% with (randInitializeWeights.m)
initial_Theta = cell(num_hl + 1, 1);      % creating the array that will embed all initial_Theta matrices
initial_Theta{1} = randInitializeWeights(input_layer_size, hl_size);
initial_nn_params = [initial_Theta{1}(:)];
for i = 2:num_hl
  initial_Theta{i} = randInitializeWeights(hl_size, hl_size); %% A Modifier
  initial_nn_params = [initial_nn_params(:); initial_Theta{i}(:)];    %% A Modifier
endfor
initial_Theta{num_hl + 1} = randInitializeWeights(hl_size, num_labels);
initial_nn_params = [initial_nn_params(:); initial_Theta{num_hl + 1}(:)];

%% ============+++ Part 4: Implement Regularization - OPTIONNAL +++===============
%% Check gradients by running checkNNGradients
% lambda = 0;
% checkNNGradients_mhl(lambda, num_hl);

%% Also output the costFunction debugging values
% debug_J  = nnCostFunction_mhl(nn_params, num_hl, input_layer_size, ...
  %                        hidden_layer_size, num_labels, X, y, lambda);

%% =================== Part 5: Training NN ===================
%% To train the neural network, "fmincg" is used, which
%% is a function which works similarly to "fminunc_mhl". Recall that these
%% advanced optimizers are able to train the cost functions efficiently as
%% long as we provide them with the gradient computations.

options = optimset('MaxIter', Max_iter);

costFunction = @(p) nnCostFunction_mhl(p, input_layer_size, ...
                                   hl_size, num_labels, X, y, lambda, num_hl);
                                 
[nn_params, cost] = fmincg_mhl(costFunction, initial_nn_params, options);

%% Obtain Thetas back from nn_params
Theta = cell(num_hl + 1, 1);                    % initialize the Theta array that will embed the various thetas of A1 and the hidden layers
nnp_start = 1;                                  % start point of the reshape function
nnp_end =  hl_size * (input_layer_size + 1);    % end point of the reshape function
nnp_s1 =  hl_size;                              % defining the size of the thetas of first layer
nnp_s2 = (input_layer_size + 1);                % defining the size of the thetas of first layer
Theta{1} = reshape(nn_params(nnp_start:nnp_end), nnp_s1, nnp_s2);       % extracting Theta_1 from the nn_params vector

nnp_s1 = hl_size;           % defining the size of the thetas of hidden layers
nnp_s2 = (hl_size + 1);     % defining the size of the thetas of hidden layers
for nhl = 2:num_hl          % Loop to exctract hidden layers from nn_params vector
  nnp_start = 1 + nnp_end;  % start point of the reshape function
  nnp_end = nnp_start - 1 + (hl_size * (hl_size + 1));     % end point of the reshape function 
  Theta{nhl} = reshape(nn_params(nnp_start:nnp_end), nnp_s1, nnp_s2);       % extracting Theta_x for the hidden layers from the nn_params vector
endfor

nnp_start = 1 + nnp_end;        % start point of the reshape function
nnp_s1 =  num_labels;           % defining the size of the thetas of first layer
nnp_s2 = (hl_size + 1);         % defining the size of the thetas of first layer
Theta{num_hl + 1} = reshape(nn_params(nnp_start:end), nnp_s1, nnp_s2);       % extracting Theta_x for the hidden layers from the nn_params vector

%% ==============+++ Part 6: Visualize Weights - OPTIONNAL +++=================
%% "Visualizing" what the neural network is learning by 
%% displaying the hidden units to see what features they are capturing in 
%% the data.

% displayData(Theta{num_hl +1} :, 2:end));


%% ================= Part 10: Implement Predict =================
%% After training the neural network, the system is assessing its accuracy
%% It computes the accuracy of the NN for the training set and fo the cross-checking set

pred = predict_mhl(Theta, num_hl, X); 
fprintf('Training Accuracy of the NN : %f\n', mean(double(pred == y)) * 100);

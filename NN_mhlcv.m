%% Neural Network with multi-hidden layers 
%% by Jean Dessain - 22 May 2020
%% This file is a personal development from the CS229 Standford class exercise
%% for neural network with 2 or more hidden layers of the same size
%% The original exercise was dedicated to single hidden layer neural network

%% =========== Part 1: Initialization  ===================
% clear ; close all; clc
fprintf('Initializing ...\n')

%% +++ Setup the parameters for defining the neural network - Input required +++
data_file = 'dataExtR_BGW.mat';     % choose the dataset for training and cross-validation
input_layer_size  = 400;            % original - 20x20 Input Images of Digits
num_hl = 3;                         % number of hidden layers in the network + initial layer X = A1 and final layer - min = 2
hl_size = 25;                       % number of units per hidden layer
num_labels = 10;                    % 10 labels, from 1 to JDE 11 as some images are not numbers   
train_size = 10000;                 % the training set can be equal or smaller than total size of the matrix X
cv_size_init = 10001;               % the test set start at position test_size_init that can be bigger than 1 => reducing the test
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
 
m = size(X, 1);                     % size of X as complete matrix 
if m < train_size                   % constraining the size of the training set
   train_size = m;
endif
if m < cv_size_init                   % constraining the size of the cross-validation set
   cv_size_init = m - 100;
endif
%% Define X for training and for test 

X_train = X(1:train_size , :);      % define part of X used for training set
y_train = y(1:train_size , 1);      % define part of y used for training set
m_train = size(X_train, 1);         % define size of X used for raining set

X_test = X(cv_size_init:end , :);   % define part of X for cross_validation, starting from initial point
y_test = y(cv_size_init:end , 1);   % define part of y for cross_validation
m_test = size(X_test, 1);             % define size of X used for cross_validation

%% ============== Part 3 : Visualization of a sample of X - OPTIONAL ==============
%% +++  Visualisation of a subset of X - OPTIONAL +++
%% Randomly select 100 data points to display the type of figures analyzed
% fprintf('Visualizing Data ...\n')
% sel = randperm(size(X, 1));
% sel = sel(1:100);
% displayData(X(sel, :));


%% ================ Part 4 : Initializing Parameters ================

initial_Theta = cell(num_hl + 1, 1);      % creating the array that will embed all initial_Theta matrices
initial_Theta{1} = randInitializeWeights(input_layer_size, hl_size);
initial_nn_params = [initial_Theta{1}(:)];
for i = 2:num_hl
  initial_Theta{i} = randInitializeWeights(hl_size, hl_size); %% A Modifier
  initial_nn_params = [initial_nn_params(:); initial_Theta{i}(:)];    %% A Modifier
endfor
initial_Theta{num_hl + 1} = randInitializeWeights(hl_size, num_labels);
initial_nn_params = [initial_nn_params(:); initial_Theta{num_hl + 1}(:)];

%% =============== Part 5: Implement Regularization - OPTIONAL ===============
%%  Once your backpropagation implementation is correct, you should now
%%  continue to implement the regularization with the cost and gradient.

%%  Check gradients by running checkNNGradients
% lambda = 0;
% checkNNGradients_2hl(lambda);

%% Also output the costFunction debugging values
% debug_J  = nnCostFunction_mhl(nn_params, num_hl, input_layer_size, ...
  %                        hidden_layer_size, num_labels, X, y, lambda);

% fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
     %    '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);

%% =================== Part 6: Training NN ===================
%% use of a cost function adapted for multi-hidden-layers network
%% call a Pollck-Ribiere optimisation function fmincg adapted for multi-hidden-layers network

options = optimset('MaxIter', Max_iter);
costFunction = @(p) nnCostFunction_mhl(p, input_layer_size, ...
                                   hl_size, num_labels, X_train, y_train, lambda, num_hl);

[nn_params, cost] = fmincg_mhl(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta = cell(num_hl + 1, 1);                    % initialize the Theta array that will embed the various thetas of A1 and the hidden layers
nnp_start = 1;                                  % start point of the reshape function
nnp_end =  hl_size * (input_layer_size + 1);    % end point of the reshape function
nnp_s1 =  hl_size;                              % defining the size of the thetas of first layer
nnp_s2 = (input_layer_size + 1);                % defining the size of the thetas of first layer
Theta{1} = reshape(nn_params(nnp_start:nnp_end), nnp_s1, nnp_s2);       % extracting Theta_1 from the nn_params vector

nnp_s1 = hl_size;           % defining the size 1 of the thetas of hidden layers
nnp_s2 = (hl_size + 1);     % defining the size 2 of the thetas of hidden layers
for nhl = 2:num_hl          % Loop to exctract hidden layers from nn_params vector
  nnp_start = 1 + nnp_end;  % start point of the reshape function
  nnp_end = nnp_start - 1 + (hl_size * (hl_size + 1));     % end point of the reshape function 
  Theta{nhl} = reshape(nn_params(nnp_start:nnp_end), nnp_s1, nnp_s2);       % extracting Theta_x for the hidden layers from the nn_params vector
endfor

nnp_start = 1 + nnp_end;        % start point of the reshape function
nnp_s1 =  num_labels;           % defining the size of the thetas of first layer
nnp_s2 = (hl_size + 1);         % defining the size of the thetas of first layer
Theta{num_hl + 1} = reshape(nn_params(nnp_start:end), nnp_s1, nnp_s2);       % extracting Theta_x for the hidden layers from the nn_params vector

%% ================= Part 7: Adequacy computation of the NN =================

pred = predict_mhl(Theta, num_hl, X_train);   
fprintf('\nTraining Accuracy: %f\n', mean(double(pred == y_train)) * 100);

pred_test = predict_mhl(Theta, num_hl, X_test);
fprintf('\nCross-checking Accuracy: %f\n', mean(double(pred_test == y_test)) * 100);


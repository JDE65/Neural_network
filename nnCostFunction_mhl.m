function [J grad] = nnCostFunction_mhl(nn_params, ...
                                   input_layer_size, hl_size, ...
                                   num_labels, X, y, lambda, num_hl)
%%  NNCOSTFUNCTION implements the neural network cost function for a multi-layer
%%  neural network which performs classification
%%  by Jean Dessain - 22 May 2020
%%  This file is a personal development from the CS229 Standford class exercise
%%  The parameters for the neural network are "unrolled" into the vector
%%  nn_params and need to be converted back into the weight matrices. 
%%  The returned parameter grad should be a "unrolled" vector of the
%%  partial derivatives of the neural network.

nnp_start = 1;                                  % start point of the reshape function
nnp_end =  hl_size * (input_layer_size + 1);    % end point of the reshape function
nnp_s1 =  hl_size;                              % defining the size of the thetas of first layer
nnp_s2 = (input_layer_size + 1);                % defining the size of the thetas of first layer
Theta{1} = reshape(nn_params(nnp_start:nnp_end), nnp_s1, nnp_s2);       % extracting Theta_1 from the nn_params vector
nnp_s1 = hl_size;           % defining the size of the thetas of hidden layers
nnp_s2 = (hl_size + 1);     % defining the size of the thetas of hidden layers
for i = 2:num_hl;           % Loop to exctract hidden layers from nn_params vector
  nnp_start = 1 + nnp_end;  % start point of the reshape function
  nnp_end = nnp_start - 1 + (hl_size * (hl_size + 1));     % end point of the reshape function 
  Theta{i} = reshape(nn_params(nnp_start:nnp_end), nnp_s1, nnp_s2);       % extracting Theta_x for the hidden layers from the nn_params vector
endfor

nnp_start = 1 + nnp_end;        % start point of the reshape function
nnp_s1 =  num_labels;           % defining the size of the thetas of first layer
nnp_s2 = (hl_size + 1);         % defining the size of the thetas of first layer
Theta{num_hl + 1} = reshape(nn_params(nnp_start:end), nnp_s1, nnp_s2);       % extracting Theta_x for the hidden layers from the nn_params vector

m = size(X, 1);
J = 0;
Theta_grad = cell(num_hl + 1, 1);                % initialize the Theta_grad array that will embed the various theta_grad matrices of A1 and the hidden layers
Theta_grad{1} = zeros(size(Theta{1}));
for i = 2:num_hl
  Theta_grad{i} = zeros(size(Theta{i}));
endfor
Theta_grad{num_hl + 1} = zeros(size(Theta{num_hl + 1}));

%% Part 1
Iden = eye(num_labels);
Ye = zeros(m, num_labels);
for i=1:m
  Ye(i, :)= Iden(y(i), :);
endfor

A = cell(num_hl + 2, 1);                % initialize the A array that will embed the various A matrices
Z = cell(num_hl + 2, 1);                % initialize the Z array that will embed the various Z matrices

A{1} = [ones(m, 1) X];
Z{2} = A{1} * Theta{1}';
for i = 2:(num_hl+1)
  A{i} = [ones(size(Z{i}, 1), 1) sigmoid(Z{i})];
  Z{i + 1} = A{i} * Theta{i}';
endfor

A{num_hl + 2} = sigmoid(Z{num_hl + 2});     % final A
H = A{num_hl + 2};

RegulJ = 0;
for i = 1:(num_hl + 1)
  RegulJ = RegulJ + (lambda / (2 * m)) * (sum(sum(Theta{i}(:, 2:end).^2, 2)));  
endfor

J = sum(sum((-Ye).*log(H) - (1-Ye).*log(1-H), 2)) / m + RegulJ;

i = 0;
ii = 0;
delt = cell(num_hl + 2, 1);
delt{num_hl + 2} = H - Ye;

for i = 1:num_hl
  ii = (num_hl + 2) - i;
  Inter = [ones(size(Z{ii}, 1), 1) Z{ii}];  
  delt{ii} = (delt{ii + 1} * Theta{ii} .* sigmoidGradient(Inter))(:, 2:end);
endfor

Delta = cell(num_hl + 1, 1);
for i = 1:(num_hl + 1)
  Delta{i} = delt{i + 1}' * A{i};
endfor

for i = 1:(num_hl + 1);
  Theta_grad{i} = Delta{i}./m + (lambda/m)*[zeros(size(Theta{i},1), 1) Theta{i}(:, 2:end)];
endfor

grad = [Theta_grad{1}(:)];
for i = 2:num_hl + 1
  grad = [grad; Theta_grad{i}(:)];
endfor

end

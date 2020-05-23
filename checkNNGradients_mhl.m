function checkNNGradients_mhl(lambda, num_hl)
%% CHECKNNGRADIENTS Creates a small neural network to check the
%% backpropagation gradients
%% CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
%% backpropagation gradients, it will output the analytical gradients
%% produced by your backprop code and the numerical gradients (computed
%% using computeNumericalGradient). These two gradient computations should
%% result in very similar values.


if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

input_layer_size = 3;
hl_size = 5;
num_labels = 3;
m = 10;

% We generate some 'random' test data

Theta = cell(num_hl + 1, 1);      % creating the array that will embed all initial_Theta matrices
Theta{1} = debugInitializeWeights(input_layer_size, hl_size);
nn_params = [Theta{1}(:)];
for i = 2:num_hl
  Theta{i} = debugInitializeWeights(hl_size, hl_size); %% A Modifier
  nn_params = [nn_params(:); Theta{i}(:)];    %% A Modifier
endfor
Theta{num_hl + 1} = debugInitializeWeights(hl_size, num_labels);
nn_params = [nn_params(:); Theta{num_hl + 1}(:)];


%% Reusing debugInitializeWeights to generate X
X = debugInitializeWeights(m, input_layer_size - 1);
y = 1 + mod(1:m, num_labels)';

%% Short hand for cost function
costFunc = @(p) nnCostFunction_mhl(p, input_layer_size, hl_size, ...
                               num_labels, X, y, lambda, num_hl);

[cost, grad] = costFunc(nn_params);
numgrad = computeNumericalGradient(costFunc, nn_params);

%% Visually examine the two gradient computations.  The two columns
%% you get should be very similar. 
%% disp([numgrad grad]);
% fprintf(['The above two columns you get should be very similar.\n' ...
  %       '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

%% Evaluate the norm of the difference between two solutions.  
%% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
%% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

% fprintf(['If your backpropagation implementation is correct, then \n' ...
  %       'the relative difference will be small (less than 1e-9). \n' ...
   %      '\nRelative Difference: %g\n'], diff);
fprintf(['If your backpropagation implementation is correct, then \n' ...
         'the relative difference will be small (less than 1e-9). \n' ...
         '\nRelative Difference: %g\n'], diff);
   
end

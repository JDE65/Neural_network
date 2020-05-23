function g = sigmoid(z)
%% S IGMOID Compute sigmoid functoon
%  J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));
end

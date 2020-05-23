function p = predict_mhl(Theta, num_hl, X)
%% PREDICT Predict the label of an input given a trained neural network
%%  by Jean Dessain - 22 May 2020
%%  This file is a personal development from the CS229 Standford class exercise

%% p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%% trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta{num_hl + 1}, 1);

p = zeros(size(X, 1), 1);

h = cell(num_hl + 1, 1);
h{1} = sigmoid([ones(m, 1) X] * Theta{1}');

for i = 2:num_hl + 1
  h_temp = [ones(m, 1) h{i - 1}];
  theta_temp = Theta{i};
  h{i} = sigmoid(h_temp * theta_temp');

endfor

[dummy, p] = max(h{num_hl + 1}, [], 2);

end

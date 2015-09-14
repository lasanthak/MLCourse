function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); % number of samples
h = size(Theta1, 1); % number of hidden units
k = size(Theta2, 1); % number of labels

% You need to return the following variables correctly 
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1: h x n+1
% Theta2: k x h+1

% add 1 to X, a1: m x n+1
a1 = [ones(m, 1) X];

% a2: m x h+1
a2 = a1 * transpose(Theta1);
a2 = 1 ./ (1 + exp(-a2));
a2 = [ones(m, 1) a2];

% values: m x k
values = a2 * transpose(Theta2);
[val, p] = max(values, [], 2);
% =========================================================================


end

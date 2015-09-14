function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X * theta);

yt = transpose(y);

thetareg = theta;
thetareg(1) = 0;

J = (1/m) * ( -yt * log(h) - (1-yt) * log(1-h) ) + (lambda/(2*m)) * (transpose(thetareg) * thetareg);

grad = (1/m) * ( transpose(X) * (h - y) ) + (lambda/m) * thetareg;


% =============================================================

end

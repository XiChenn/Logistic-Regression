function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); 
J = 0;
grad = zeros(size(theta)); 

h = sigmoid(X * theta); % 1*m

J = ((log(h') * y) +  log(1 - h') * (1 - y)) / -m + (lambda / (2 * m)) * (theta' * theta - theta(1) ^2);  
%theta' * theta - theta(0) ^2 - not regularize theta(0);
    
gradWithoutReg = X' * (h - y) ./ m;
grad = gradWithoutReg + (lambda / m) .* theta;

grad(1) = gradWithoutReg(1);

end

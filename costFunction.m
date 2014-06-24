function [J, grad] = costFunction(initial_theta, X, y)

m = size(X, 1);
J = 0;
grad = zeros(size(initial_theta)); 

h = sigmoid(X * initial_theta); % 1*m

J = ((log(h') * y) +  log(1 - h') * (1 - y)) / -m;
    
grad = X' * (h - y) ./ m;

end
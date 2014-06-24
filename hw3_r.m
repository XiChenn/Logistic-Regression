%% Regularized Logistic Regression
%
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contain the X values and the third column
%  contains the label (y).

data = load('regdata.txt');
X = data(:, [1, 2]); y = data(:, 3);


% Use the mapFeature function to create all the polynomial features and the extra column of 1s
X = mapFeature(X(:,1), X(:,2));

% Initialize theta
initial_theta = zeros(size(X, 2), 1);

% Set lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

% Train!

% Initialize theta
initial_theta = zeros(size(X, 2), 1);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% predict the accuracy on the training set 
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

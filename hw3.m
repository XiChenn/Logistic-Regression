%%Logistic regression
%%Adapted from the Stanford Machine Learning course

%% Initialization
clear ; close all; clc

%% Load Data
%  Hw, Attendance, Exams, Grade, Pass/Fail
%  contains the label.

data = load('Grades2.txt');

%%Initialize X and Y. For the first experiment, use HW and Attendance
%X = data(:, 1: size(data, 2) - 1);
X = data(:, [1:2]);
y = data(:, size(data, 2));
%% ==================== Part 1: Plotting ====================
%  Plot the data

plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Homework')
ylabel('Attendance')

% Specified in plot order
legend('Passed', 'Failed')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% Cost and Gradient for Logistic Regression 

%  Set up your data matrix to have the column of ones for the Intercept term
X = [ones(size(X, 1), 1), X];


% Create an initial theta of all zeros
initial_theta = zeros(size(X, 2), 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta: %f\n', cost);
fprintf('Gradient at initial theta: \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% Optimize Theta

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Use fminunc to compute theta and cost. 
[theta, cost] = ...
fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta 
fprintf('Cost at optimized theta: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);


fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Predicting Outcomes
%  The parameters can now be used to make predictions. You'll want to use theta and the sigmoid 
% function to make the prediction 
%
% Predict the propability that a student with a 75 on their homework and a 55 for attendance
% pass the course
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%

prob = sigmoid([1, 45, 85] * theta); % use 45 on homework and 85 on attendance as in "Lab5_HW3.pdf"
fprintf(['For a student with scores 45 and 85, we predict an pass ' ...
            'probability of %f\n\n'], prob);

            

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


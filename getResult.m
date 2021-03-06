clear; close all; clc;

File = cell(10, 1);
File{1} = 'datasets/ionosphere_train.mat';
File{2} = 'datasets/ionosphere_test.mat';
File{3} = 'datasets/isolet_train.mat';
File{4} = 'datasets/isolet_test.mat';
File{5} = 'datasets/liver_train.mat';
File{6} = 'datasets/liver_test.mat';
File{7} = 'datasets/mnist_train.mat';
File{8} = 'datasets/mnist_test.mat';
File{9} = 'datasets/mushroom_train.mat';
File{10} = 'datasets/mushroom_test.mat';

for test_i = 1: 5
%set parameters for neural network model
test_case = 2 * test_i - 1;   %1, 3, 5, 7, 9
max_iter = 10000;
error_thres = 0.05;
step = 0.2;
lambda = 0;


learning_data = File{test_case};
fprintf('Neural Network Model\n');
fprintf('Test case is %s\n', learning_data);

load(learning_data);
[N, D] = size(X);    #D: dimension of X
[N, K] = size(Y);    #K: dimension of Y

%add intercept term to X
X = [ones(N, 1) X];

train_num = int32(4 * N / 5);
validation_num = double(N - train_num);

%%%%%%%%% initial cost %%%%%%%%%%
% init_W = randomWeights(2, D);
% init_V = randomWeights(K, 2);
% neural_param = [init_W(:); init_V(:)];
% [init_cost, init_gradient] = neuralCostFunction(neural_param, 2, X, Y, 0);
% fprintf('Cost at initial weights (random from -0.01 to 0.01): %f\n', init_cost);

%%%%%%%%% find best number of hidden units %%%%%%%%%%
error = zeros(1, 20);
for H = 1 : 20 %20
  %find minimum H
  err = 0;
  for cnt = 1 : 10 %10
    %%%%%%%%% split dataset into training data and validation data %%%%%%%%%%
    %shuffle X by rows
    order = randperm(N);
    X_rand = X(order, :);
    Y_rand = Y(order, :);

    train_X = X_rand(1 : train_num, :);
    valid_X = X_rand(train_num + 1 : end, :);
    train_Y = Y_rand(1 : train_num, :);
    valid_Y = Y_rand(train_num + 1 : end, :);

    %inital weight matrix
    W = randomWeights(H, D);
    V = randomWeights(K, H);
    init_param = [W(:); V(:)];

    %train system with train data
    % options = optimset('MaxIter', 500);
    % options = optimset('GradObj', 'on', 'MaxIter', 1000);
    % costFunction = @(p) neuralCostFunction(p, H, train_X, train_Y, lambda);
    % [neuralParam, cost] = fmincg(costFunction, init_param, options);
    % [neuralParam, cost] = fminunc(costFunction, init_param, options);

    cost = 100;
    iter = 0;
    while (cost > error_thres) && (iter < max_iter)
      [cost gradient] = neuralCostFunction(init_param, H, train_X, train_Y, lambda);
      dW = reshape(gradient(1 : (D + 1) * H), H, D + 1);
      dV = reshape(gradient(((D + 1) * H) + 1 : end), K, H + 1);
      W = W + step * dW;
      V = V + step * dV;
      init_param = [W(:); V(:)];
      iter = iter + 1;
    end
    % fprintf('After %d iterations, cost = %f\n', iter, cost);

    %cross validation
    valid_Z = sigmoid(W * valid_X');  %%%%%%%%%
    valid_Z = [ones(1, validation_num); valid_Z];
    Y_esti = sigmoid(V * valid_Z);
    err = err - 1 / double(validation_num) * sum(valid_Y .* log(Y_esti') + (1 - valid_Y) .* log(1 - Y_esti'));

    % for c = 1 : validation_num
    %   testX = valid_X(c, :);
    %   testZ = sigmoid(W * testX');
    %   testZ = [1; testZ];
    %   testY = sigmoid(V * testZ);
    %   trueY = valid_Y(c, 1);
    %   fprintf('units %d: testY = %f, trueY ＝ %d\n', H, testY, trueY);
    %   err = err - trueY * log(testY) - (1 - trueY) * log(1 - testY);
    % end
  end
  error(1, H) = err / 10;
end

[min_error, H] = min(error(:));
fprintf('\nFind minimum hidden units number is %d.\n', H);

fprintf('\nRe-train model with whole data.....\n');
lambda = 0;
cost = 100;
iter = 0;
%inital weight matrix
W = randomWeights(H, D);
V = randomWeights(K, H);
init_param = [W(:); V(:)];
t = cputime;
while (cost > error_thres) && (iter < max_iter)
  [cost gradient] = neuralCostFunction(init_param, H, X, Y, lambda);
  dW = reshape(gradient(1 : (D + 1) * H), H, D + 1);
  dV = reshape(gradient(((D + 1) * H) + 1 : end), K, H + 1);
  W = W + step * dW;
  V = V + step * dV;
  init_param = [W(:); V(:)];
  iter = iter + 1;
end
e = cputime - t;
fprintf('CPUtime without I/O for task is %.2f sec.\n', e);
fprintf('Final cost of re-train data is %f\n after %d iterations\n', cost, iter);


fprintf('\nTest accuracy of training data...\n');
test_Z = sigmoid(W * X');  %%%%%%%%%
test_Z = [ones(1, N); test_Z];
Y_cal = sigmoid(V * test_Z);
%Train data accuracy
accuracy = 0;
boundary = 0.5 * ones(size(Y_cal));
class1 = Y_cal > boundary;
class2 = Y_cal <= boundary;
accuracy = accuracy + sum(class1 .* Y') + sum(class2 .* (1 - Y'));
% for i = 1 : N
%   if (Y_cal(i) > 0.5 && Y(i) == 1)
%     accuracy = accuracy + 1;
%   end
%   if (Y_cal(i) <= 0.5 && Y(i) == 0)
%     accuracy = accuracy + 1;
%   end
% end
accuracy = accuracy / N;
fprintf('Training data %d correct in %d, accuracy is %.2f%%\n\n', accuracy * N, N, accuracy * 100);


fprintf('\nTest accuracy of test data...\n');
test_data = File{test_case + 1};
load(test_data);
test_num = size(X, 1);
X = [ones(test_num, 1), X];

test_Z = sigmoid(W * X');  %%%%%%%%%
test_Z = [ones(1, test_num); test_Z];
Y_cal = sigmoid(V * test_Z);
%Test data accuracy
accuracy = 0;
boundary = 0.5 * ones(size(Y_cal));
class1 = Y_cal > boundary;
class2 = Y_cal <= boundary;
accuracy = accuracy + sum(class1 .* Y') + sum(class2 .* (1 - Y'));
% for i = 1 : test_num
%   if (Y_cal(i) > 0.5 && Y(i) == 1)
%     accuracy = accuracy + 1;
%   end
%   if (Y_cal(i) <= 0.5 && Y(i) == 0)
%     accuracy = accuracy + 1;
%   end
% end
accuracy = accuracy / test_num;
fprintf('Test data %d correct in %d, accuracy is %.2f%%\n\n', accuracy * test_num, test_num, accuracy * 100);
fprintf('\n\n');
end

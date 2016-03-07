function [cost, gradient] = binaryCostFunction(weight, X, Y)
  train_num = length(Y);

  cost = 0;
  gradient = zeros(size(weight));
  hypothesis = sigmoid(X * weight);
  cost = (- 1 / train_num) * sum(Y .* log(hypothesis) + (1 - Y) .* log(1 - hypothesis));

  #compute gradient
  for i = 1 : train_num
    gradient = gradient + ((hypothesis(i) - Y(i)) * X(i, :))';
  end
  gradient = (1 / train_num) * gradient;
end

function [cost gradient] = neuralCostFunction(neuralParam, hidden_dim, X, Y, lambda)

[N, X_dim] = size(X);
[N, Y_dim] = size(Y);

W = reshape(neuralParam(1 : X_dim * hidden_dim), hidden_dim, X_dim);
V = reshape(neuralParam((1 + X_dim * hidden_dim) : end), Y_dim, hidden_dim + 1);

Z = sigmoid(W * X');  %%%%%%%%%
Z = [ones(1, N); Z];
Y_esti = sigmoid(V * Z);


cost = (- 1 / N) * sum(Y .* log(Y_esti') + (ones(size(Y)) - Y) .* log(ones(size(Y)) - Y_esti'));

% for stable
stable_tag = (sum(sum(W(:, 2:end) .^ 2)) + sum(sum(V(:, 2:end) .^ 2))) * lambda / (2 * N);
cost = cost + stable_tag;


%%%%%%%%% compute gradients of W and V %%%%%%%%%
grad_W = zeros(size(W));
grad_V = zeros(size(V));

% for i = 1 : N
%   xi = X(i, :)';	%D x 1
%   zi = W * xi;		%H x 1
%   zi = sigmoid(zi);
%   zi = [1; zi];
%   yi = V * zi;		%K x 1
%   yi = sigmoid(yi);
%
%   err_y = Y(i, :) - yi;		%K x 1
%   err_z = (grad_V' * err_y)(2:end, 1) .* zi(2:end, 1) .* (1 - zi(2:end, 1));
%
%   grad_V = grad_V + err_y * zi';
%   grad_W = grad_W + err_z * xi';
% end
Zh = W * X';
Zh = sigmoid(Zh);
Zh = [ones(1, N); Zh];
Y_esti = V * Zh;
Y_esti = sigmoid(Y_esti);
err_Y = Y - Y_esti';
err_Z = (V' * err_Y')(2:end, :) .* Zh(2:end, :) .* (ones(size(Zh(2:end, :))) - Zh(2:end, :)); %H x N

grad_V = err_Y' * Zh';
grad_W = err_Z * X;

dW = grad_W / N + lambda * [zeros(hidden_dim, 1) W(:, 2 : end)] / N;
dV = grad_V / N + lambda * [zeros(Y_dim, 1) V(:, 2 : end)] / N;

gradient = [dW(:); dV(:)];

end

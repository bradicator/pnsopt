function [nll,g,H,HH] = pnsopt_logitloss(w,X,y,weights,downsample)
% Negative log likelihood for binary logistic regression
% w: d*1
% X: n*d
% y: n*1, should be -1 or 1


[n, d] = size(X);
if nargin < 4, weights = ones(n,1); end
if nargin < 5, downsample = floor(n/6); end

y01 = (y+1)/2;
mu = sigmoid(X*w);
mu = max(mu, eps); % bound away from 0
mu = min(1-eps, mu); % bound away from 1
nll = -sum(weights .* (y01 .* log(mu) + (1-y01) .* log(1-mu)));
Xw = X .* repmat(colvec(weights), 1, d);
if nargout > 1
  g = Xw'*(mu-y01);
end

if nargout == 3
  H = diag(sqrt(mu.*(1-mu)))*X * sqrt(weights(1));
end

if nargout == 4
  index = randsample(1:n, downsample);
  mu = mu(index);
  HH = X(index,:)' * diag(mu.*(1-mu))* X(index,:) * sqrt(weights(1));
  H = 0; % place holder for H
end

end
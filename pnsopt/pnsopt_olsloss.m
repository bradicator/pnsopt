function [nll,g,H,HH] = pnsopt_olsloss(w,X,y,weights,downsample)
% OLS loss function 0.5*||Xw-y||_2^2
% w: d*1
% X: n*d
% y: n*1, should be -1 or 1


[n, d] = size(X);
if nargin < 4, weights = ones(n,1); end
if nargin < 5, downsample = floor(n/6); end

mu = X * w - y;
nll = mu' * mu / 2;
Xw = X .* repmat(colvec(weights), 1, d);
if nargout > 1
  g = Xw'*mu;
end

if nargout == 3
  H = Xw'*X;
end

if nargout == 4
  index = randsample(1:n, downsample);
  mu = mu(index);
  HH = Xw(index,:)' * X(index,:);
  H = 0; % place holder for H
end

end
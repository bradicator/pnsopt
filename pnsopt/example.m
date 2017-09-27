% generate some data with binary outcome
rng(5);
n = 100;
p = 300;
s = 10;
X = randn(n, p);
beta0 = [ones(s, 1); zeros(p-s, 1)];
y = X * beta0;
prob = exp(y) ./ (1 + exp(y));
y = random('bino', 1, prob);
y = y*2 - 1;

% define smoothF and nonsmoothF
lambda = 1;
l1_pen  = prox_l1(lambda);
logistic_obj = @(w) pnsopt_logitloss(w,X,y);
ols_obj = @(w) pnsopt_olsloss(w,X,y);

% initial point
w0 = zeros(p,1);

% fit lasso
[ w1, f1, output1 ] = pnsopt(ols_obj, l1_pen, w0);
[ w2, f2, output2 ] = tfocs( ols_obj, [],l1_pen, w0);


% fit logistic regression
[ w3, f3, output3 ] = pnsopt(logistic_obj, l1_pen, w0);
[ w4, f4, output4 ] = tfocs( logistic_obj, [],l1_pen, w0);

rng(5);
n = 100;
p = 500;
s = 10;
X = randn(n, p);
beta0 = [ones(s, 1); zeros(p-s, 1)];
y = X * beta0;
prob = exp(y) ./ (1 + exp(y));
y = random('bino', 1, prob);
y = y*2 - 1;

lambda = 0.5;
l1_pen  = prox_l1(lambda);
logistic_obj = @(w) pnsopt_logitloss(w,X,y);

w0 = zeros(p,1);
[ w, f, output ] = pnsopt(logistic_obj, l1_pen, w0);
tic;
[ w2, f2, output2 ] = tfocs( logistic_obj, [],l1_pen, w0);
toc;
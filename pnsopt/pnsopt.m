function [ x, f, output ] = pnsopt( smoothF, nonsmoothF, x, options )
% pnopt : Proximal Newton-type methods
% 
% [ x, f, output ] = pnsopt( smoothF, nonsmoothF, x ) starts at x and seeks a 
%   minimizer of the objective function in composite form. 
% 
% smoothF is a handle to a function that takes one input and returns
% according to nargs. If nargs=2, return function value and gradient. If
% nargs=3, return function value, gradient, hessian square root. If
% nargs=4 and downsampling is used, return function value, gradient, 
% a placeholder for hessian square root, downsampled hessian. see the
% design of pnopt_logitloss.m for example.
%                                                                      
% nonsmoothF is a handle to a function that takes one input and returns h(x)
% and prox(x).
%  
% [ x, f, output ] = pnsopt( smoothF, nonsmoothF, x, options ) replaces the default
%   optimization parameters with those in options, a structure created using the
%   pnopt_optimset function.
% 
% $Date: 2017/08/21 $
 
% ============ Process options ============
  
  tfocs_opts = struct(...
    'alg'        , 'GRA' ,... % supports GRA only now
    'maxIts'     , 500   ,...
    'printEvery' , 0     ,...
    'restart'    , 100   ...
    );
  
% default options
  if exist( 'tfocs', 'file')
    default_options = pnsopt_optimset(...
      'debug'          , 0          ,... % debug mode 
      'desc_param'     , 0.0001     ,... % sufficient descent parameter
      'display'        , 1         ,... % display frequency (<= 0 for no display) 
      'max_fun_evals'  , 5000       ,... % max number of function evaluations
      'max_iter'       , 500        ,... % max number of iterations
      'subprob_solver' , 'tfocs'    ,... % solver for solving subproblems
      'tfocs_opts'     , tfocs_opts ,... % subproblem solver options
      'ftol'           , 1e-8       ,... % stopping tolerance on relative change in the objective function 
      'optim_tol'      , 1e-6       ,... % stopping tolerance on optimality condition
      'xtol'           , 1e-8       ,... % stopping tolerance on solution
      'sketch_dim'     , 100        ,... % sketch dimension
      'theta'          , 0.85       ,... % stopping tolerance
      'beta'           , 0.1        ,... % hessian approx tolerance
      'delta'          , 0.001      ,... % amount of I added for p.d.
      'backtrack'      , 1          ,... % 1 means yes
      'exact_stopping' , 0          ,... % is the inverse of H used? very expensive if so.
      'downsample'     , 0          ,... % gaussian sketching or down sample
      'method'         , 'sketch'   ...  % sketch, downsample, or bfgs
      );
    
  else
    error('please install the tfocs on my github page')
  end
  
  if nargin > 3
    if isfield( options, 'tfocs_opts' )
      options.tfocs_opts = merge_struct( tfocs_opts, options.tfocs_opts );
    end
    options = pnsopt_optimset( default_options, options );
  else
    options = default_options;
  end
  
  % ============ Call solver ============
  
 
 [ x, f, output ] = pnsopt_pqn( smoothF, nonsmoothF, x, options );
 
  
  
function S3 = merge_struct( S1 ,S2 )
% merge_struct : merge two structures
%   self-explanatory 
% S2 has higher priority in name clash
  S3 = S1;
  S3_names = fieldnames( S2 );
  for k = 1:length( S3_names )
    if isfield( S3, S3_names{k} )
      if isstruct( S3.(S3_names{k}) )
        S3.(S3_names{k}) = merge_struct( S3.(S3_names{k}),...
          S2.(S3_names{k}) );
      else
        S3.(S3_names{k}) = S2.(S3_names{k});
      end
    else
      S3.(S3_names{k}) = S2.(S3_names{k});
    end
end
  
  
function [ x, f_x, output ] = pnsopt_pqn( smoothF, nonsmoothF, x, options )
% pnsopt_pqn : Proximal Newton Sketching and other quasi newton methods
% 
% [ x, f, output ] = pnopt_pqn( smoothF, nonsmoothF, x, options ) starts at x and 
%   seeks a minimizer of the objective function in composite form. smoothF is a 
%   handle to a function that returns the smooth function value and gradient. 
%   nonsmoothF is a handle to a function that returns the nonsmooth function 
%   value and proximal mapping. options is a structure created using the 
%   pnopt_optimset function.
% 
  REVISION = '$Revision: 0.9.2$';
  DATE     = '$Date: Aug. 21, 2017$';
  REVISION = REVISION(11:end-1);
  DATE     = DATE(8:end-1);
% ============ Process options ============
  
  debug          = options.debug;
  desc_param     = options.desc_param;
  display        = options.display;
  max_fun_evals  = options.max_fun_evals;
  max_iter       = options.max_iter;
  subprob_solver = options.subprob_solver;
  ftol           = options.ftol;
  optim_tol      = options.optim_tol;
  xtol           = options.xtol;
  sketch_dim     = options.sketch_dim;
  theta          = options.theta;
  beta           = options.beta;
  delta          = options.delta;
  backtrack      = options.backtrack;
  exact_stopping = options.exact_stopping;
  tfocs_opts     = options.tfocs_opts;
  downsample     = options.downsample;
  method         = options.method;
 
% ============ Initialize variables ============
  
  pnsopt_flags
  
  iter         = 0; 
  loop         = 1;
  x_dim        = size(x, 1);
  
  trace.f_x    = zeros( max_iter + 1, 1 );
  trace.optim  = zeros( max_iter + 1, 1 );
  trace.time  = zeros( max_iter + 1, 1 );
  trace.tfocsiter  = zeros( max_iter + 1, 1 );
  trace.sparseiter = zeros( max_iter + 1, 1 );
  trace.fun_evals  = zeros( max_iter + 1, 1 );
  
  if display > 0  
    fprintf( ' %s\n', repmat( '=', 1, 64 ) );
    fprintf( '                  PNSKETCH v.%s (%s)\n', REVISION, DATE );
    fprintf( ' %s\n', repmat( '=', 1, 64 ) );
    fprintf( '%4s   %12s  %12s  %4s %12s \n',...
       '','Obj. val.', 'v-hnorm', 'tfocs_iter', 'time' );
  end
  

  global r
  r = 0;
  
% ------------ Evaluate objective function at starting x ------------
  
  [ g_x, grad_g_x ] = smoothF( x );
    h_x         = nonsmoothF( x );
    f_x         = g_x + h_x;
% ------------ Start collecting data for display and output ------------
  

  [ ~, x_prox ] = nonsmoothF( x - grad_g_x, 1 );
    optim       = norm( x_prox - x, 'inf' );
  
  trace.f_x(1)    = f_x;
  trace.optim(1)  = optim;
  trace.time(1) = 0;
  trace.tfocsiter(1) = 0;
  trace.fun_evals(1) = 1;
  
  if display > 0
    fprintf( ' %4d |  %12.4e  %12.4e  %4d %12.2f \n', ...
      iter, f_x, optim, 0, 0);
  end
  
% ------------ Check if starting x is optimal ------------
  
  if optim <= optim_tol
    flag    = FLAG_OPTIM;
    message = MESSAGE_OPTIM;
    loop    = 0;
  end

% ============ Main Loop ============
  tic;
  timepoint = 0;
  while loop
    iter = iter + 1; 
    
    
    
  % ------------ Construct Hessian approximation ------------
    
    switch method
        
        case 'downsample'
            [g_x, grad_g_x, ~, H_x] = smoothF(x);
            H_x = H_x + delta*eye(x_dim);
        case 'sketch'
            [g_x, grad_g_x, G_root] = smoothF(x);
            S = normrnd(0, 1, [sketch_dim, size(G_root, 1)]);
            SG = S * G_root;
            %H_x = G_root' * G_root + delta*eye(x_dim); % for debug
            H_x = SG' * SG / sketch_dim + delta*eye(x_dim);
        case 'bfgs'
            if iter > 1
                s    =  x - x_old;
                y    = grad_g_x - grad_g_old;
                qty1 = cholB' * ( cholB * s );
                if s'*y > 1e-9
                    cholB = cholupdate( cholupdate( cholB, y / sqrt( y' *s ) ), qty1 / ...
                    sqrt( s' * qty1 ), '-' );
                end
                H_x = cholB' * ( cholB );
            else
                cholB = eye( length( x ) );
                H_x = 0.5 * eye( length( x ) );
            end
            [g_x, grad_g_x] = smoothF(x);
    end
    
    
    h_x         = nonsmoothF( x );
    f_x         = g_x + h_x;
    
    % use which stopping rule? |v|_H or |v|_2
    if exact_stopping
        % use pinv incase delta = 0. can be slower than inv though.
        Hinv = pinv(H_x);
    else
        Hinv = 0; % pass in a placeholder
    end
    
  % ------------ Solve subproblem for a search direction ------------
    

      quadF = @(z) pnsopt_quad( SG, sketch_dim, grad_g_x, delta, g_x, z - x );
      
      xstart = x;
      tfocs_opts.stopFcn = @(f, x, x_old, L) pns_stop(x,x_old,...
          L,H_x,theta,Hinv,xstart,exact_stopping);


      [ x_prox, subf, ~ ] = ...
        tfocs( quadF, [], nonsmoothF, x, tfocs_opts );

     
      tfocsiter = subf.niter;
      p = x_prox - x;
      
  % ------------ Conduct line search ------------
    
    x_old      = x;
    f_old      = f_x; %#ok<NASGU>
    h_old      = nonsmoothF(x_old);
    grad_g_old = grad_g_x;
    
    v_hnorm = sqrt(p' * H_x * p);
    
    if backtrack
        kappa = 0.5;
        rho = desc_param;
        bt_max_iter = 100;
        bt_iter = 0;
        alpha = 1;
        while 1
            bt_iter = 1 + bt_iter;
            y = x_old + alpha * p;
            [g_y, ~] = smoothF(y);
            f_y = g_y + nonsmoothF(y);
            if f_y < f_old + alpha * rho * v_hnorm^2
                x = x_old + alpha * p;
                %disp('sucess')
                break
            elseif bt_iter > bt_max_iter
                disp('backtrack went wrong');
                break
            end
            alpha = kappa * alpha;
        end
        trace.fun_evals(iter+1) = trace.fun_evals(iter+1) + bt_iter;
    else
        if v_hnorm > 0.09
            alpha = theta/(1+theta*v_hnorm/sqrt(1-beta));
            x = x_old + alpha * p;
        else
            x = x_old + p;
        end
    end
    
    [f_x, grad_g_x, ~] = smoothF(x);
    f_x = f_x + nonsmoothF(x); 
  % ------------ Collect data for display and output ------------
    
    timepoint = toc;
 
    optim     = v_hnorm;
    
    trace.f_x(iter+1)        = f_x; 
    trace.optim(iter+1)      = optim;
    trace.time(iter+1)       = timepoint;
    trace.tfocsiter(iter+1)  = tfocsiter;
    trace.sparseiter(iter+1)  = sum(abs(x) > 1e-7);
    trace.fun_evals(iter+1) = trace.fun_evals(iter+1) + 1;
    
    
    if display > 0 && mod( iter, display ) == 0
      fprintf(  ' %4d |  %12.4e  %12.4e  %4d  %12.2f\n', ...
      iter, f_x, optim, tfocsiter, timepoint);
    end
    
    pnsopt_stop
    
  end
  
% ============ Clean up and exit ============
  
  trace.f_x        = trace.f_x(1:iter+1);
  trace.optim      = trace.optim(1:iter+1);
  trace.time      = trace.time(1:iter+1);
  trace.tfocsiter = trace.tfocsiter(1:iter+1);
  trace.sparseiter = trace.sparseiter(1:iter+1);
  trace.fun_evals = trace.fun_evals(1:iter+1);
  trace.fun_evals = cumsum(trace.fun_evals);
  
  if display > 0 
    fprintf(  ' %4d |  %12.4e  %12.4e  %4d  %12.2f\n', ...
      iter, f_x, optim, tfocsiter, timepoint);
    fprintf( ' %s\n', repmat( '-', 1, 64 ) );
  end
  
  output = struct( ...    
    'flag'       , flag       ,...
    'msg'        , message    ,...
    'iters'      , iter       ,...
    'optim'      , optim      ,...
    'options'    , options    ,...
    'trace'      , trace       ...
    );
  
  
  
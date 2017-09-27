
    if optim <= optim_tol
      flag    = FLAG_OPTIM;
      message = MESSAGE_OPTIM;
      loop    = 0;
      %disp('a');
    elseif norm( x - x_old, 'inf' ) / max( 1, norm( x_old, 'inf' ) ) <= xtol 
      flag    = FLAG_XTOL;
      message = MESSAGE_XTOL;
      loop    = 0;
      %disp('b');
    elseif abs( f_old - f_x ) / max( 1, abs( f_old ) ) <= ftol
      flag    = FLAG_FTOL;
      message = MESSAGE_FTOL;
      loop    = 0;
      %disp('c');
    elseif iter >= max_iter 
      flag    = FLAG_MAXITER;
      message = MESSAGE_MAXITER;
      loop    = 0;
    end
    
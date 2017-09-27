function stop = pns_stop(x,x_old,L,H,theta,Hinv,xstart, exact)
  global r;
  r = H * (x - x_old) - L * (x - x_old);
  v_hnorm = sqrt((x-xstart)' * H * (x-xstart));
  
  % use which stopping rule? |v|_H or |v|_2
  if exact
    r_hnorm = sqrt(r' * Hinv * r);
  else
    % the coef downhere is somewhat important
    r_hnorm = 0.05 * norm(r);
  end
  stop = r_hnorm < (1-theta)*v_hnorm;
  
     
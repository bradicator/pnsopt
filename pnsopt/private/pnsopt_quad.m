function varargout = pnsopt_quad( SG, sketch_dim, q, delta, r, x )
  
  xx = SG * x;
  grad_f_y = SG' * xx / sketch_dim + delta * x + q;
  varargout{1} = 0.5 * x' * ( grad_f_y + q ) + r;
  
  if nargout > 1
    varargout{2} = grad_f_y;
  end
  
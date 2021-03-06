function options = pnsopt_optimset( varargin )
% pnopt_optimset : Set options for PNS
%
% options = pns_optimset( 'param1', val1, 'param2', val2,... ) creates an options
%   structure in which the named parameters have the specified values. Default
%   values are used for parameters that are not specified.
%   
%
%   $Revision: 0.9.0 $  $Date: 2017/09/01 $
% 
  % Print out possible values of options. outdated. need updates
  if nargin == 0 && nargout == 0
    fprintf('See Default parameters for PNSOPT in pnsopt.m:\n');
    return;
  end

  Names = [
    'debug          '
    'desc_param     '
    'display        '
    'max_fun_evals  '
    'max_iter       '
    'subprob_solver '
    'tfocs_opts     '
    'ftol           '
    'optim_tol      '
    'xtol           '
    'sketch_dim     '
    'theta          '
    'beta           '
    'delta          '
    'backtrack      '
    'exact_stopping '
    'downsample     '     
    'method         '
    ];
  
  [m,n] = size(Names); 
  names = lower(Names);

  % Combine all leading options structures o1, o2, ... in l1Set(o1,o2,...).
  options = [];
%   for j = 1:m
%     eval(['options.' Names(j,:) '= [];']);
%   end
  i = 1;
  while i <= nargin
    arg = varargin{i};
    if ischar(arg), break; end
    if ~isempty(arg)                      % [] is a valid options argument
      if ~isa(arg,'struct')
        error(sprintf(['Expected argument %d to be a string parameter name ' ...
          'or an options structure\ncreated with pnopt_optimset.'], i)); %#ok<SPERR>
      end
      for j = 1:m
        if any(strcmp(fieldnames(arg),deblank(Names(j,:))))
          eval(['val = arg.' Names(j,:) ';']);
        else
          val = [];
        end
        if ~isempty(val)
          eval(['options.' Names(j,:) '= val;']);
        end
      end
    end
    i = i + 1;
  end
  
  % A finite state machine to parse name-value pairs.
  if rem(nargin-i+1,2) ~= 0
    error('Arguments must occur in name-value pairs.');
  end
  expectval = 0;                          % start expecting a name, not a value
  while i <= nargin
    arg = varargin{i};

    if ~expectval
      if ~ischar(arg)
         error('Expected argument %d to be a string parameter name.', i);
      end

      lowArg = lower(arg);
      j = strmatch(lowArg,names);
      if isempty(j)                       % if no matches
        error('Unrecognized parameter name ''%s''.', arg);
      elseif length(j) > 1                % if more than one match
        % Check for any exact matches (in case any names are subsets of others)
        k = strmatch(lowArg,names,'exact');
        if length(k) == 1
          j = k;
        else
          msg = sprintf('Ambiguous parameter name ''%s'' ', arg);
          msg = [msg '(' deblank(Names(j(1),:))]; %#ok<AGROW>
          for k = j(2:length(j))'
            msg = [msg ', ' deblank(Names(k,:))]; %#ok<AGROW>
          end
          error('%s).', msg);
        end
      end
      expectval = 1;                      % we expect a value next

    else
      eval(['options.' Names(j,:) '= arg;']);
      expectval = 0;
      
    end
    i = i + 1;
  end

  if expectval
    error('Expected value for parameter ''%s''.', arg);
  end


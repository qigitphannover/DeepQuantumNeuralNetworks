%Computes all Cost functions in Generalization Task

function  CList = GeneralizationTask(phi_in,phi_out,U,M,varargin)

dim = 2^M(1);
%size(phi_in,2) = N_NumTrain can be bigger than dim

%Don't change, default is false
if_variable_t = false;
if_variable_n = false;
%if ~isempty(varargin)
    if rem(length(varargin), 2) == 1 % test that optional input is paired
        error('Provide the optional input arguments as name/value pairs.')
    end
    varargin = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);

    if isfield(varargin, 'lambda') % get momentum variable_t
        %update t to given value
        lambda = varargin.('lambda');
    else
        error('No lambda given')
    end

    if isfield(varargin, 'iter') % get momentum variable_t
        %update t to given value
        iter = varargin.('iter');
    else
        error('No iter given')
    end

    if isfield(varargin, 'variable_t') % get momentum variable_t
        %update t to given value
        t = varargin.('variable_t');
        %n_default fix the n while n varying
        n_default = varargin.('n_default');
        if_variable_t = true;
    elseif isfield(varargin, 'variable_n') % get variable_n
        %update n to given value
        n = varargin.('variable_n');
        %t_default fix the t while n varying
        t_default = varargin.('t_default');
        if_variable_n = true;
    else
        fprintf('Time t and n fixed.')
        if_variable_t = false;
        if_variable_n = false;
    end

%variable t and n is not allowed (need to be implemented)
if( if_variable_t & if_variable_n )
    error('Variable_t and variable_n is not allowed. Please choose just one variable parameter of these. Otherwise implement it.')
end

CList= zeros(1,dim);
for j = 1:dim
    fprintf('\nNumber of training pairs = %d', j)
    phi_in_use = phi_in(:,1:j);
    phi_out_use = phi_out(:,1:j);

    %Specify good iteration number and lambda
    if(if_variable_t)
        [x,z] = TrainNetwork(phi_in_use,phi_out_use,U,M, lambda, iter, 'variable_t', t, 'n_default', n_default);
        CList(j) = CostNetwork(phi_in,phi_out,x,M, 'variable_t', t, 'n_default', n_default);
    elseif(if_variable_n)
        [x,z] = TrainNetwork(phi_in_use,phi_out_use,U,M, lambda, iter, 'variable_n', n, 't_default', t_default);
        CList(j) = CostNetwork(phi_in,phi_out,x,M, 'variable_n', n, 't_default', t_default);
    else
        [x,z] = TrainNetwork(phi_in_use,phi_out_use,U,M, lambda, iter);
        CList(j) = CostNetwork(phi_in,phi_out,x,M);
    end

end

end

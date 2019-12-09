%Calculates Costfunctions for Noisy data Plot. phi_in_start and
%phi_out_start should be RandomUnitaryTrainingData

function CList = Noisydata(phi_in_start,phi_out_start,U,M,s,varargin)
%s is the number of steps between each point

N_NumTrain = size(phi_in_start,2);

dim = 2^M(1);

%Don't change, default is false
if_variable_t = false;
if_variable_n = false;
%if ~isempty(varargin)
    if rem(length(varargin), 2) == 1 % test that optional input is paired
        error('Provide the optional input arguments as name/value pairs.')
    end
    varargin = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);

    if isfield(varargin, 'lambda') % get lambda
        %update t to given value
        lambda = varargin.('lambda');
    else
        error('No lambda given')
    end

    if isfield(varargin, 'iter') % get iter
        %update t to given value
        iter = varargin.('iter');
    else
        error('No iter given')
    end

    if isfield(varargin, 'variable_t') % get variable_t
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

N_iter = N_NumTrain/s;
% N_iter should be integer

CList = zeros(1,N_iter);
phi_in = phi_in_start;
phi_out = phi_out_start;
[phi_in_noisy,phi_out_noisy] = Randomtrainingdata(N_NumTrain,dim);

%variable t and n is not allowed (need to be implemented)
if( if_variable_t & if_variable_n )
    error('Variable_t and variable_n is not allowed. Please choose just one variable parameter of these. Otherwise implement it.')
end

%fprintf('\n  default_n = %1.3f \n', n_default)
%fprintf('\n  variable_t = %1.3f \n', t)
%fprintf('\n  lambda = %1.3f \n', lambda)
%fprintf('\n  iter = %1.3f \n', iter)

for j = 1:N_iter+1
    fprintf('\n Number of noisy pairs = %d %d',s, j)
    if(if_variable_t)
        %%New with error inside NN
        x = TrainNetwork(phi_in,phi_out,U,M, lambda, iter, 'variable_t', t, 'n_default', n_default);
        CList(j) = CostNetwork(phi_in_start,phi_out_start,x,M, 'variable_t', t, 'n_default', n_default);
    elseif(if_variable_n)
        %%New with error inside NN
        x = TrainNetwork(phi_in,phi_out,U,M, lambda, iter, 'variable_n', n, 't_default', t_default);
        CList(j) = CostNetwork(phi_in_start,phi_out_start,x,M, 'variable_n', n, 't_default', t_default);
    else
        %%Old without error inside NN
        x = TrainNetwork(phi_in,phi_out,U,M, lambda, iter);
        CList(j) = CostNetwork(phi_in_start,phi_out_start,x,M);
    end

    if j < N_iter+1
        List = randperm(N_NumTrain);
        phi_in = phi_in_start(:,List);
        phi_out = phi_out_start(:,List);

        List = randperm(N_NumTrain);
        phi_in_noisy = phi_in_noisy(:,List);
        phi_out_noisy = phi_out_noisy(:,List);

        phi_in(:,1:s*j) = phi_in_noisy(:,1:s*j);
        phi_out(:,1:s*j) = phi_out_noisy(:,1:s*j);
    end

end



end

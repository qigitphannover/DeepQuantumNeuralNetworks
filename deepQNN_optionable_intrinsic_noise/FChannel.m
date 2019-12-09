%Applicaton of F (conjugate) Channel
function rho_result = FChannel(rho_start,U,N_StartLay,N_ResLay,varargin)

%This function maps the start density matrix rho_start to the image under the F
%channel, which is rho_result.

%N_StartLay: Number of neurons in the starting layer where rho_start lives.
%N_ResLay: Number of neurons in the resulting layer where rho_result lives.

%Dont chage if_variable_t or n, it has to be false as default for above lying functions
if_variable_t = false;
if_variable_n = false;

if ~isempty(varargin)
    if rem(length(varargin), 2) == 1 % test that optional input is paired
        error('Provide the optional input arguments as name/value pairs.')
    end
    varargin = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);
    if isfield(varargin, 'variable_t') % get variable_time
        %update t to given value
        t = varargin.('variable_t');
        %n_default fix the n while n varying
        n_default = varargin.('n_default');
        if_variable_t = true;
    end
    if isfield(varargin, 'variable_n')% get variable_n
        %update n to given value
        n = varargin.('variable_n');
        %t_default fix the t while n varying
        t_default = varargin.('t_default');
        if_variable_n = true;
    end
end


RHO_whole = kron(eye(2^N_ResLay),rho_start);

for j = 1 : N_StartLay
    j_1 = N_StartLay - j +1;

    %Take number of rows or columns, here rows where used
    size_of_U = size(U(:,:,j_1),1);

    if(if_variable_t)
        V = Swap(kron(RandomUnitary1(size_of_U,t, n_default) * U(:,:,j_1) * RandomUnitary1(size_of_U,t, n_default),eye(2^(N_StartLay - 1))),[N_ResLay+1,N_ResLay+j_1],2*ones(1,N_StartLay+N_ResLay));
    elseif(if_variable_n)
        V = Swap(kron(RandomUnitary1(size_of_U, t_default, n) * U(:,:,j_1) * RandomUnitary1(size_of_U, t_default, n),eye(2^(N_StartLay - 1))),[N_ResLay+1,N_ResLay+j_1],2*ones(1,N_StartLay+N_ResLay));
    else
        V = Swap(kron(U(:,:,j_1),eye(2^(N_StartLay - 1))),[N_ResLay+1,N_ResLay+j_1],2*ones(1,N_StartLay+N_ResLay));
    end

    RHO_whole = V'*RHO_whole*V;
end
RHO_whole = kron(eye(2^N_ResLay),[1;zeros(2^N_StartLay-1,1)]*[1;zeros(2^N_StartLay-1,1)]')*RHO_whole;
rho_result = PartialTrace(RHO_whole,N_ResLay+1:N_ResLay+N_StartLay,2*ones(1,N_StartLay+N_ResLay));
end

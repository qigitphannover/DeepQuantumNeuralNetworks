%Calculates Cost function of Network given a set of input and target output
%states
function C = CostNetwork(phi_in,phi_out,U,M, varargin)

%Don't change, default is false
if_variable_t = false;
if_variable_n = false;


if ~isempty(varargin)
    if rem(length(varargin), 2) == 1 % test that optional input is paired
        error('Provide the optional input arguments as name/value pairs.')
    end
    varargin = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);

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
end


%phi_in (phi_out): array with columns being the input (output) vectors
N_NumTrain = size(phi_in,2);
C = 0;

for x=1:N_NumTrain
    if(if_variable_t)
        C = C + dot(phi_out(:,x),ApplyNetwork(phi_in(:,x),U,M, 'variable_t', t, 'n_default', n_default)*phi_out(:,x));
    elseif(if_variable_n)
        C = C + dot(phi_out(:,x),ApplyNetwork(phi_in(:,x),U,M, 'variable_n', n, 't_default', t_default)*phi_out(:,x));
    else
        C = C + dot(phi_out(:,x),ApplyNetwork(phi_in(:,x),U,M)*phi_out(:,x));
    end

end

C = real(C/N_NumTrain);

end

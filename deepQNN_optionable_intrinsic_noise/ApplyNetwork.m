%Calculates the output state/density matrix for som given input vector
%phi_in
function rho_out = ApplyNetwork(phi_in,U,M,varargin)

% M array of Number of Neurons in each layer, i.e. size(M,2) = Num_Layers, M(1,j) = Num_Neurons in Layer j

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


N_Layers = size(M,2);

rho_out = phi_in*phi_in';

for k = 2:N_Layers
    if(if_variable_t)
       rho_out = ApplyLayer(rho_out,U{k},M(k),M(k-1), 'variable_t', t, 'n_default', n_default);
    elseif(if_variable_n)
       rho_out = ApplyLayer(rho_out,U{k},M(k),M(k-1), 'variable_n', n, 't_default', t_default);
    else
       rho_out = ApplyLayer(rho_out,U{k},M(k),M(k-1));
    end
end


end

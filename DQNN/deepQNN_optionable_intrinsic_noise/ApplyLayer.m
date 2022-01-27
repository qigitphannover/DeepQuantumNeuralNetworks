%Application of all unitaries in one given layer. This corresponds to the
%action of the \mathcal{E} on a given start density matrix rho_in.
function rho_out = ApplyLayer(rho_in,U,N_CorrLay,N_PrevLay,varargin)

%Don't change, default is false
if_variable_t = false;
if_variable_n = false;

%pho_in (pho_out): input (output) state of the corresponding layer
%N_CorrLay  (N_PrevLay): Number of neurons in the corresponding (previous) Layer
%U array containing all N_CorrLay Unitaries of the corresponding Layer


%Initialising Input State on all N_CorrLay + N_PrevLay Qubits
RHO_in_whole = kron(rho_in,[1;zeros(2^N_CorrLay -1,1)]*[1;zeros(2^N_CorrLay -1,1)]');

%Calculating Output on all N_CorrLay + N_PrevLay Qubits
RHO_out_whole = RHO_in_whole;

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

for j= 1:N_CorrLay
    %RandomUnitary1(dim,t,n) instead random to add noise to ML
    if(if_variable_t)
        newRand = RandomUnitary1(2^(N_PrevLay+1), t, n_default)*U(:,:,j)*RandomUnitary1(2^(N_PrevLay+1), t, n_default);
    elseif(if_variable_n)
        newRand = RandomUnitary1(2^(N_PrevLay+1), t_default, n)*U(:,:,j)*RandomUnitary1(2^(N_PrevLay+1), t_default, n);
    else
    %if variable_t and variable_n are false, no noise to the neurons added
        newRand = U(:,:,j);
    end

    V = Swap(kron(newRand,...
    eye(2^(N_CorrLay - 1))),[N_PrevLay+1,N_PrevLay+j],2*ones(1,N_CorrLay+N_PrevLay));
    RHO_out_whole = V*RHO_out_whole*V';
end

%Calculating Output state of the Neurons in the corresponding Layer
rho_out = PartialTrace(RHO_out_whole,1:N_PrevLay,2*ones(1,N_CorrLay+N_PrevLay));

end

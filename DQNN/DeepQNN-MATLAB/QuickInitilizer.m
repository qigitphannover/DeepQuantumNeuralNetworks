function U = QuickInitilizer(M, varargin)
% This functions quickly initializes a set of random unitaries for the
% network with configuration M. 
%
% optional in as name/value pairs (function(..., 'name', value, ...)):
% sparse: cell array S defining the sparsity of the neural network;
%         for the k-th layer in M, the j-th neuron in this layer, 
%         and its connection to the i-th neuron in the (k-1)-th layer, 
%         S{k-1}(j,i) is either zero or one, where 1 means connected;
%         default is a fully connected network
% out:
% The unitaries for each layer are stored in the cell array U.
% U{k}(:,:,j) is the unitary belonging to the jth neuron in the kth layer.
% If sparse, U{k}(:,:,j) acts as an identity on disconnected qubits.

if rem(length(varargin), 2) == 1 % test that optional input is paired
    error('Provide the optional input arguments as name/value pairs.')
end
varargin = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);

N_NumLay = size(M,2);

if ~isfield(varargin, 'sparse') % fully connected
    for k = 2:N_NumLay
        for j = 1:M(k)
            U{k}(:,:,j) = Randomunitary(2^(M(k-1)+1));
        end
    end
else
    sparsity = varargin.('sparse'); % sparse
    for k = 2:N_NumLay
        for j = 1:M(k)
            NumBonds = M(k-1); % number of connections if fully connected
            bonds = 1:NumBonds;
            cut = bonds(~sparsity{k-1}(j,:)); % indices of connections to cut
            NumCut = length(cut); % number of connections to cut
            % random unitary acting on (NumBonds-NumCut+1) qubits
            % tensored with identities on NumCut qubits:
            Ukj = kron(Randomunitary(2^(NumBonds-NumCut+1)), eye(2^NumCut)); 
            for i = 1:NumCut % swap identities to disconnected qubits
                Ukj = Swap(Ukj, [cut(i), NumBonds+2-i], 2*ones(1, NumBonds+1));
            end 
            U{k}(:,:,j) = Ukj;
        end
    end
end

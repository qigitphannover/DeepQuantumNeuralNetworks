function [mean_fidelity, var_fidelity] = FlipAnalytics(psi, p)
% FlipAnalytics provides properties of spin-flip noise on an arbitrary state
%
% in:
% psi: state in the tensor product basis
% p: spin-flip probability per qubit or array of spin-flip probabilities
% out:
% mean_fidelity: mean fidelity of correctly weighted flipped psi with psi (for each p)
% var_fidelity: variance of the fidelity (for each p)

psi = psi(:);
p = p(:);
dim = length(psi); % dimension of Hilbert space
m = log2(dim); % number of qubits

% all possible flip combinations binaries2(i,:)
% 1: flip, 0: no flip
binaries1 = dec2bin(0:(dim-1), m); 
binaries2 = reshape(str2num(binaries1(:)), dim, m);

k = sum(binaries2, 2)'; % flip numbers k(i)
probs = p.^k .* (1-p).^(m-k); % flip probabilities probs(:,i)

flip1 = [0, 1; 1, 0]; % spin flip for one qubit
id = eye(2);
flips = zeros(dim, dim, dim); % flip operators flips(:,:,i)
for s = 1:dim
    b = binaries2(s,:);
    if b(1) == 0
        f = id;
    else
        f = flip1;
    end
    for r = 2:m
        if b(r) == 0
            f = kron(f, id);
        else
            f = kron(f, flip1);
        end
    end
    flips(:,:,s) = f;
end

psi_flipped = squeeze(multiprod(flips, psi)); % flipped states psi_flipped(:,i)
fidelities = abs(psi' * psi_flipped).^2; % fidelities(i) of flipped states with original state

mean_fidelity = sum(probs .* fidelities, 2);
var_fidelity = sum(probs .* (fidelities - mean_fidelity).^2, 2);
end
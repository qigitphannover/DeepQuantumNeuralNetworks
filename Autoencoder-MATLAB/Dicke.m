function psi = Dicke(m, k)
% Dicke creates Dicke states
%
% in:
% m: number of qubits
% k: number of up spins
% out:
% psi: m-qubit Dicke state with k up spins,
%      i.e. the symmetrization of |1>^k |0>^(m-k),
%      in the tensor product basis
% Dicke(3, 1) is the W-state
% Dicke(m, 1) is the generalized W-state

n = nchoosek(m,k);
binaries = zeros([n, m]); 
% rows: binary vectors for all distinct permutations

perms = nchoosek(1:m, k); % column indeces of the ones for each permutation
rows = kron(ones(1, k), 1:n);
cols = perms(:)';

binaries(sub2ind([n, m], rows, cols)) = 1;

binaries2 = 1 - fliplr(binaries); 
% MATLAB binary numbers: [1, 0] = 1, [0, 1] = 2 etc.
% the flip is optional when summing over permutations
% bitwise inversion b --> 1-b because canonically:
% |1> ~ (1, 0) ~ small index in tensor product state,
% |0> ~ (0, 1) ~ large index in tensor product state

% each permutation contributes one 1 to psi
ind = bi2de(binaries2) + 1; % indeces of the ones 

psi = zeros(2^m, 1);
psi(ind) = 1;
psi = psi / sqrt(sum(psi));
end
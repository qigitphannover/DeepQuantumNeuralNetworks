function psi = Cluster(Gamma)
% Cluster creates cluster states
%
% in:
% Gamma: adjacency matrix, 
%        neighbors: 1; else, including diagonal: 0
% out:
% psi: m-qubit cluster state with adjacency matrix Gamma \in {0,1}^(m x m),
%      sum_{a \in {0,1}^m} (-1)^(a^T*Gamma*a/2) |a>,
%      0 in a corresponds to [1,0], 1 to [0,1], 
%      joint eigenstates of K_j with eigenvalue 1,
%      K_j: sigma_x on jth qubit, sigma_z on neighbors,
%      in the tensor product basis

m = length(Gamma); % number of qubits
d = 2^m;
ind = 1:d;

% binary basis (rows)
% 0 ~ [1,0], 1 ~ [0,1] 
% ordered according to tensor product basis
binaries1 = dec2bin(ind-1, m); 
binaries2 = reshape(str2num(binaries1(:)), d, m);

sign = diag(binaries2 * Gamma * binaries2') / 2;
psi = (-1).^sign; 
psi = psi / sqrt(d);
end
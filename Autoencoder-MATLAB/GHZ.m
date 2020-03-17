function psi = GHZ(m, phi)
% GHZ creates ghz states with relative phases
%
% in: 
% m: number of qubits
% phi (optional, default = 0): array of relative phases
% out:
% psi: psi(:,j) is the state (up + exp(i*phi(j)) down) / sqrt(2) 
%      in the tensor product basis

if nargin < 2
    phi = 0;
end

psi = zeros(2^m, length(phi));
psi(1,:) = 1;
psi(end,:) = exp(1i*phi);
psi = psi / sqrt(2);
end

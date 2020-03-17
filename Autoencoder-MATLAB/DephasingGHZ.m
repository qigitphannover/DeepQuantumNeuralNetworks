function GHZ_noisy = DephasingGHZ(m, n, p, mean_phi)
% Dephasing applies n times dephasing noise to an m-qubit GHZ state
% with variance 2*pi*p.
%
% in:
% m: number of qubits
% phi (optional, default = 0): array of relative phases
% n: number of noise realizations
% p: dephasing probability per qubit, maximal physical value is 0.5
% out:
% GHZ_noisy: GHZ_noisy(:,i) is the ith realization of a noisy psi

if nargin < 4
    mean_phi = 0;
end
phases = mean_phi + randn(n,1)*pi*p;%/sqrt(m);

GHZ_noisy = GHZ(m,phases);

end

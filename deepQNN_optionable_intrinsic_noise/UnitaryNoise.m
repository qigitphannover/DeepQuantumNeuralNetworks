function psi_noisy = UnitaryNoise(psi, n, t, q)
% SpinFlipNoise applies n times spin-flip noise to a state psi
% with noise parameter t and number of noisy gates q.
%
% in:
% psi: noiseless state.
% n: number of noise realizations.
% t: noise strength, "time" during which noise acts, around t=1 the noise is
% so strong the distribution is essentially uniform.
% q: number of noisy gates, controls how good of an approximation to
% Brownian the noise is.
% out:
% psi_noisy: psi_noisy(:,i) is the ith realization of a noisy psi.

dim =  length(psi); % dimension of Hilbert space
psi_noisy = zeros(dim, n);
for i = 1:n
    psi_noisy(:,i) = RandomUnitary1(dim,t,q)*psi;
end
function [mean_fidelity, var_fidelity] = FlipAnalyticsDicke(m, k, p)
% FlipAnalyticsDicke provides properties of spin-flip noise on Dicke states
%
% in:
% m: number of qubits
% k: number of up spins
% p: spin-flip probability per qubit or array of spin-flip probabilities
% out:
% mean_fidelity: mean Dicke fidelity (for each p)
% var_fidelity: variance of Dicke fidelity (for each p)

p = p(:);
if k > floor(m/2) % exploit symmetry between Dicke(m,k) and Dicke(m,m-k)
    k = m - k;
end
flips = 0:m;
ind = 0:k;

probs = arrayfun(@(x) nchoosek(m, x), flips) .* p.^flips .* (1-p).^(m-flips);
fids = zeros(1, m+1);
finfids = (arrayfun(@(x) nchoosek(2*x, x) * nchoosek(m - 2*x, k - x), ind) / nchoosek(m, k)).^2;
fids(2 * ind + 1) = finfids;

mean_fidelity = sum(probs .* fids, 2);
var_fidelity = sum(probs .* (fids - mean_fidelity).^2, 2);
end
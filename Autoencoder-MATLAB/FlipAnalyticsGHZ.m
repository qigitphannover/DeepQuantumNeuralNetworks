function [mean_fidelity, var_fidelity] = FlipAnalyticsGHZ(m, p)
% FlipAnalyticsGHZ provides properties of spin-flip noise on GHZ states
%
% in:
% m: number of qubits
% p: spin-flip probability per qubit or array of spin-flip probabilities
% out:
% mean_fidelity: mean GHZ fidelity (for each p)
% var_fidelity: variance of GHZ fidelity (for each p)

mean_fidelity = p.^m + (1 - p).^m ; % probability of zero effective flips
var_fidelity = mean_fidelity .* (1 - mean_fidelity);

% denoising is expected if zero effective flips is most probable
% this is true for all p < 1/2
% for p = 1/2, any effective flipped state has the same probability
end
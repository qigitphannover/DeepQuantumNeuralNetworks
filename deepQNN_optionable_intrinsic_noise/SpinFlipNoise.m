function psi_noisy = SpinFlipNoise(psi, n, p)
% SpinFlipNoise applies n times spin-flip noise to an m-qubit state psi
% with spin-flip probability p per qubit
%
% in:
% psi: noiseless m-qubit state in tensor product basis
% n: number of noise realizations
% p: spin-flip probability per qubit, maximal physical value is 0.5
% out:
% psi_noisy: psi_noisy(:,i) is the ith realization of a noisy psi

if p > 0.5
    error('Spin-flip probability per qubit greater than 0.5 is unphysical.')
end

dim =  length(psi); % dimension of Hilbert space
m = log2(dim); % number of qubits

% define flips(:,:,i) as flip of ith spin
flip1 = [0, 1; 1, 0]; % spin flip for one qubit
id = eye(2);
flips = zeros(dim, dim, m);
for i = 1:m
    f = flip1;
    for j = 1:(i-1)
        f = kron(id, f);
    end
    for j = (i+1):m
        f = kron(f, id);
    end
    flips(:,:,i) = f;
end

% for every noise realization psi_noisy(:,i) and every qubit j
% draw random number r(j) from uniform distribution in (0, 1)
% and apply jth spin flip with probability p, i.e. if r(j) <= p 
psi_noisy = zeros(dim, n);
for i = 1:n
    r = rand(m,1); % random numbers
    pn = psi;
    for j = 1:m
        if r(j) <= p
            pn = flips(:,:,j) * pn; % apply spin flip
        end
    end
    psi_noisy(:,i) = pn;
end

end

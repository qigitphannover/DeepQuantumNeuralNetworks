function U =  BrownianCircuit(dim,n,dt)
%BrownianCircuit(dim,n,var,dt) creates a random unitary matrix by multiplying
% exp(i H_1 dt) exp(i H_2 dt) ... exp(i H_n dt)
%for random normal distributed hermitian dim-dimensional matricies \{ H_i \}_{i=1}^{n} with variance 1.

U = eye(dim);

for j= 1:n
    Re = randn(dim);
    Im = 1i*randn(dim);
    C = Re + Im;
    %for normalization from C
    H = (C + C')/4;
    %Add noise to U by calculating random walk with step dt
    U = U*expm(1i*H*dt);
end
end


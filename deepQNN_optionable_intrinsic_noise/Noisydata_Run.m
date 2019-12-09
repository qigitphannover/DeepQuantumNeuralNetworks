%Calculates Costfunctions for Noisy data Plot. phi_in_start and
%phi_out_start should be RandomUnitaryTrainingData

M=[2,3,2];
dim = 2^M(1);
%s is the number of steps between each point
s=10;
% N_iter = N_NumTrain/s should be intege, N_NumTrain = size(phi_in_start,2);

lambda = 1;
iter = 300;
t_default = 0.005;
n_default = 20;

%number of training pairs 100
[phi_in_start,phi_out_start,U] = RandomUnitaryTrainingsData(100,dim);

noisyCList = Noisydata(phi_in_start,phi_out_start,QuickInitilizer(M),M,s, 'lambda', lambda, 'iter', iter, 'variable_t', t_default, 'n_default', n_default)

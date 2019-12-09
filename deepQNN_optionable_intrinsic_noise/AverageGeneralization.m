%This function averages over 20 rounds of the Generalization task
%Use in console [x,y] = AverageGeneralization
function [z_3, z_ave] = AverageGeneralization

%Set different seed when starting matlab (important for randn)
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)))

lambda = 1.5
iter = 1000
t_default = 0.0133
n_default = 20

if_variable_t = true;
if_variable_n = false;
runs = 20;

z_3 = zeros(runs,8);
z_ave = zeros(1,8);

%Specify the right M here
M =[3,3,3];


for j =1:runs
    fprintf('\n Run = %d', j)

    %Specify how the traingsdata should be chosen here
    [phi_in,phi_out] = RandomUnitaryTrainingsData(10,8);

     U = QuickInitilizer(M);

    if(if_variable_t)
        z_3(j,:) = GeneralizationTask(phi_in,phi_out,U,M, 'lambda', lambda, 'iter', iter, 'variable_t', t_default, 'n_default', n_default);
    elseif(if_variable_n)
        z_3(j,:) = GeneralizationTask(phi_in,phi_out,U,M, 'lambda', lambda, 'iter', iter, 'variable_n', n_default, 't_default', t_default);
    else
        z_3(j,:) = GeneralizationTask(phi_in,phi_out,U,M, 'lambda', lambda, 'iter', iter);
    end

    z_ave = z_ave + z_3(j,:);

end
z_ave = z_ave/runs;

end

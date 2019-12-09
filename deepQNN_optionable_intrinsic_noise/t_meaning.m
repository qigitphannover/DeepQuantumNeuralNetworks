%This function averages over 20 rounds of the Generalization task
%Use in console [mean1,var1,mean2,var2] = t_meaning
function [mean_fidelity_average, var_fidelity_average, mean_fidelity_fidelity_average, var_fidelity_fidelity_average] = t_meaning

format long

%Set different seed when starting matlab (important for randn)
RandStream.setGlobalStream(RandStream('mt19937ar','seed',sum(100*clock)))

%set start and end time t, if if_variable_n is true and number of different time evaluation points
if_variable_t = true;
t_start = 0.0000;
t_end = 0.0500;
t_number = 101;
n_default = 20;
%set start and end time n, if if_variable_n is true and number of different time evaluation points
if_variable_n = false;
n_start = 10;
n_end = 40;
n_number = 4;
t_default = 0.025;


%For M=[3,3,3] its size= 2^3
size_of_U = 2^3;
phi_pairs = 500;
runs = 20;

if(if_variable_t)
    mean_fidelity_average = zeros(1,t_number);
    var_fidelity_average = zeros(1,t_number);

    %2nd method
    mean_fidelity_fidelity_average = zeros(1,t_number);
    var_fidelity_fidelity_average = zeros(1,t_number);
elseif(if_variable_n)
    mean_fidelity_average = zeros(1,n_number);
    var_fidelity_average = zeros(1,n_number);
else
    error('Choose if_variable_n or if_variable_n true')
end



for run_count =1:runs
    fprintf('\n Run = %d', run_count)
    %Specify how the traingsdata should be chosen here
    %just phi_in needed as U=1 (identity)
    [phi_in,phi_out] = RandomUnitaryTrainingsData(phi_pairs,size_of_U);

    if(if_variable_t)
        mean_fidelity = zeros(1,t_number);
        var_fidelity = zeros(1,t_number);
        flist = zeros(1,phi_pairs);

        mean_fidelity_fidelity = zeros(1,t_number);
        var_fidelity_fidelity = zeros(1,t_number);
        flist_fidelity = zeros(1,phi_pairs);


        t_step = (t_end - t_start)/(t_number-1);

        for t_count = 1:t_number
            t = t_start + (t_count - 1)*t_step;
            fprintf('t = %1.8f \n\n', t)

            %size_target = size(phi_in);
            %size_compare = size(K1*K2);
            %fprintf('size_target = %1.4f \n\n', size_target)
            %fprintf('size_compare = %1.4f \n\n', size_compare)

            for i = 1:phi_pairs
                K1 = RandomUnitary1(size_of_U,t, n_default);
                K2 = RandomUnitary1(size_of_U,t, n_default);

                flist_not_transposed(i) = dot(phi_in(:,i), K2*K1*phi_in(:,i));
                flist_transposed(i) = dot(phi_in(:,i), ctranspose(K1)*ctranspose(K2)*phi_in(:,i));
                flist(i) =real(flist_not_transposed(i) * flist_transposed(i));

                %2nd mehtod
                flist_fidelity(i) = Fidelity(phi_in(:,i),K2*K1*phi_in(:,i));
            end
            mean_fidelity(1,t_count) = mean(flist);
            var_fidelity(1,t_count) = var(flist);

            %2nd mehtod
            mean_fidelity_fidelity(1,t_count) = mean(flist_fidelity);
            var_fidelity_fidelity(1,t_count) = var(flist_fidelity);
        end
    else
        error('Choose if_variable_n or if_variable_n true')
    end

    %summing up the runs to calculate average later on
    mean_fidelity_average = mean_fidelity_average + mean_fidelity;
    var_fidelity_average = var_fidelity_average + var_fidelity;

    mean_fidelity_fidelity_average = mean_fidelity_fidelity_average + mean_fidelity_fidelity;
    var_fidelity_fidelity_average = var_fidelity_fidelity_average + var_fidelity_fidelity;

end
%calculate average
mean_fidelity_average = mean_fidelity_average/runs;
var_fidelity_average = var_fidelity_average/runs;

%2nd method
mean_fidelity_fidelity_average = mean_fidelity_fidelity_average/runs;
var_fidelity_fidelity_average = var_fidelity_fidelity_average/runs;

fprintf('mean_fidelity_fidelity_average = %1.9f \n\n', mean_fidelity_fidelity_average)

NAME = 't_meaning_runs20_dim8_pairs500_t0.0000to0.0500_n20_allrandomK'
dlmwrite(strcat(NAME,'_mean.csv'), mean_fidelity_average,'precision',10)
dlmwrite(strcat(NAME,'_var.csv'), var_fidelity_average,'precision',10)

%NAME_fidelity = 't_meaning_runs20_dim8_pairs10000_t0.0000to0.0133_n20_fidelity'
%dlmwrite(strcat(NAME_fidelity,'_mean.csv'), mean_fidelity_fidelity_average,'precision',10)
%dlmwrite(strcat(NAME_fidelity,'_var.csv'), var_fidelity_fidelity_average,'precision',10)


end

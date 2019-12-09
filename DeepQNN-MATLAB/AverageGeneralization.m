%This function averages over 20 rounds of the Generalization task
function [z_3,z_ave] = AverageGeneralization
z_3 = zeros(20,8);
z_ave = zeros(1,8);

%Specify the right M here
M =[3,3];

for j =1:20
    
    %Specify how the traingsdata should be chosen here
    [phi_in,phi_out] = RandomUnitaryTrainingsData(10,8);
            
     U = QuickInitilizer(M);
    
    z_3(j,:) = GeneralizationTask(phi_in,phi_out,U,M); 
    z_ave = z_ave + z_3(j,:);
end
z_ave = z_ave/20;
    
end
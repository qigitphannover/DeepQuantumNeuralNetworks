%Computes all Cost functions in Generalization Task

function  CList = GeneralizationTask(phi_in,phi_out,U,M)

dim = 2^M(1);
%size(phi_in,2) = N_NumTrain can be bigger than dim

CList= zeros(1,dim);
for j = 1:dim
   phi_in_use = phi_in(:,1:j);
   phi_out_use = phi_out(:,1:j);
   
   %Specify good iteration number and lambda
   [x,z] = TrainNetwork(phi_in_use,phi_out_use,U,M,1.5,350);
   CList(j) = CostNetwork(phi_in,phi_out,x,M);
end

end
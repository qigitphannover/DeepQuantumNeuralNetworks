%Calculates Cost function of Network given a set of input and target output
%states
function C = CostNetwork(phi_in,phi_out,U,M)

%phi_in (phi_out): array with columns being the input (output) vectors
N_NumTrain = size(phi_in,2);
C = 0;

for x=1:N_NumTrain  
 C = C + dot(phi_out(:,x),ApplyNetwork(phi_in(:,x),U,M)*phi_out(:,x));
end

C = real(C/N_NumTrain);

end
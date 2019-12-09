%Generates an array of random trainings data, where for each the output
%state is the image of the input state under a given (randomly chosenn)
%unitary
function [phi_in,phi_out,U] = RandomUnitaryTrainingsData (n,m)

phi_in = randn(m,n) + i* randn(m,n);

for j = 1:n
    phi_in(: ,j) = (phi_in(: ,j))/norm(phi_in(: ,j));
end

U = Randomunitary(m);

phi_out = zeros(m,n);

for j =1:n
   
    phi_out(:,j) = U *phi_in(:,j);
end

end
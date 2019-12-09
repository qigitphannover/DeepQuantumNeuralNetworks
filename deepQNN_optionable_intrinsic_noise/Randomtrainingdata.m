%Generates an array of random input and output vectors
function [phi_in,phi_out] = Randomtrainingdata(n,m)

phi_in = randn(m,n) + i*randn(m,n);

for j = 1:n
       phi_in(: ,j) = (phi_in(: ,j))/norm(phi_in(: ,j));
end

phi_out =  randn(m,n) + i*randn(m,n);

for j = 1:n
   phi_out(: ,j) = (phi_out(: ,j))/norm(phi_out(: ,j));
end

end
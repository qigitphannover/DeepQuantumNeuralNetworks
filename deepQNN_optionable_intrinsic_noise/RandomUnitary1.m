function U = RandomUnitary1(dim,t,n)
%RandomUnitary1(deviation) creates a dim-dimensional unitary close to identiry.
%For t=0 the answer is identity, for large enough t
%(for practical purpouses, t~1) 
%the answer is a unitary sampled from Haar distribution.
%The error to Texp(\int_0^t i \pi H(t) dt ) with random H(t) scales as 1/sqrt(n).

%2pi for scaling t from 0 to 1,
%sqrt(1/n) make the noise strength completely dependent on just t  
    U = BrownianCircuit(dim,n,sqrt(1/(n*dim))*2*pi*t);
end

%Creates a random unitary on \mathbb{C}^N
function U = Randomunitary(N)

U =  zeros(N);

U(:,1) = randn(N,1) + i*randn(N,1);
U(:,1) = U(:,1) / norm(U(:,1));
for j= 2:N

    U(:,j) = randn(N,1) + i*randn(N,1);
    for k = 1:j-1
    psi = U(:,j);
    U(:,j) = U(:,j) -  dot(U(:,k),psi)*U(:,k);
    end
    U(:,j) = U(:,j)/norm(U(:,j));
end


end


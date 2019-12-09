function range = SpectralRange(A)
%Outputs a spectral gap of a matrix A.

    range = norm(norm(A)*eye(size(A,1)) - A);
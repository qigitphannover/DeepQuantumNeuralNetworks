
function [phi_in, phi_out, test_in, test_out] = AutoData(mode, n, m)
dim = 2^m;
if mode == 1
    p = 0.4;
    n0 = round(p*n);
    phi0 = ghz(m, 0);
    flip0 = [0, 1; 1, 0];
    id = eye(2);
    flip = zeros(dim,dim,n);
    phi = zeros(dim,n);
    for i = 1:m
        f = flip0;
        for j = 1:(i-1)
            f = kron(id, f);
        end
        for j = (i+1):m
            f = kron(f, id);
        end
        flip(:,:,i) = f;
        phi(:,i) = f*phi0;
    end
    % phi_rand = phi(:,randi(m,1,3*n));
    % phi_in = phi_rand(:,1:n);
    % phi_out = phi_rand(:,n+1:2*n);
    % test_in = phi_rand(:,2*n+1:end);
    phi_rand = phi(:,randi(m,1,3*n0));
    rep_phi0 = repmat(phi0, 1, n-n0);
    phi_in = cat(2, phi_rand(:,1:n0), rep_phi0);
    phi_in = phi_in(:,randi(n,1,n));
    phi_out = cat(2, phi_rand(:,n0+1:2*n0), rep_phi0);
    phi_out = phi_out(:,randi(n,1,n));
    test_in = cat(2, phi_rand(:,2*n0+1:end), rep_phi0);
    test_out = repmat(phi0, 1, n);
end
if mode == 2
    ghz_plus = ghz(m,0);
    ghz_minus = ghz(m,pi);
    phi_plus = repmat(ghz_plus, 1, floor(n/2));
    phi_minus = repmat(ghz_minus, 1, ceil(n/2));
    phi_in = cat(2, phi_plus, phi_minus);
    phi_out = phi_in;
    test_in = cat(2, ghz_plus, ghz_minus);
    test_out = test_in;
end
if mode == 3
    sgm = 0.5;
    phi_rand = repmat(exp(1i*normrnd(0, sgm, [1,3*n])),dim,1).*ghz(m,normrnd(0, sgm, [1,3*n]));
    phi_in = phi_rand(:,1:n);
    phi_out = phi_rand(:,n+1:2*n);
    test_in = phi_rand(:,2*n+1:end);
    test_out = repmat(ghz(m,0), 1, n); 
end
end

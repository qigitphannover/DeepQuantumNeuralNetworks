function flist = Fidelity(target, compare)
% Fidelity computes the fidelity between pure target and
% pure or mixed comparison states
%
% in:
% target: target(:,i) is the ith pure target state of length > 1
% compare: compare(:,i) is the ith pure state or
%          compare(:,:,i) is the ith density matrix
%          to be compared to the ith target state
% out:
% flist: array of fidelities

st = size(target);
sc = size(compare);
n = st(end); % number of fidelities
typec = length(sc); % determine if compare is mixed or pure
if sc(end) ~= n
    if n == 1 && isequal(sc, [st(1), st(1)])
        % ensure that a single density matrix is not interpreted as two vectors
        typec = 3;
    else
        error('The last dimensions of the input arguments have to match.')
    end
end
if typec == 2 % pure-pure state fidelity
    %print('pure state: abs^2 is calculated.')
    flist = abs(dot(compare, target)).^2;
elseif typec == 3 % pure-mixed state fidelity
    flist = zeros(1,n);
    for i = 1:n
        flist(i) = real(dot(target(:,i), compare(:,:,i)*target(:,i)));
    end
else
    error('Second input argument has wrong number of dimensions.')
end
end

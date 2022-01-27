%Applicaton of F (conjugate) Channel
function rho_result = FChannel(rho_start,U,N_StartLay,N_ResLay)

%This function maps the start density matrix rho_start to the image under the F
%channel, which is rho_result.

%N_StartLay: Number of neurons in the starting layer where rho_start lives.
%N_ResLay: Number of neurons in the resulting layer where rho_result lives.

RHO_whole = kron(eye(2^N_ResLay),rho_start);

for j = 1 : N_StartLay
    j_1 = N_StartLay - j +1;
    V = Swap(kron(U(:,:,j_1),eye(2^(N_StartLay - 1))),[N_ResLay+1,N_ResLay+j_1],2*ones(1,N_StartLay+N_ResLay));
    RHO_whole = V'*RHO_whole*V;
end
RHO_whole = kron(eye(2^N_ResLay),[1;zeros(2^N_StartLay-1,1)]*[1;zeros(2^N_StartLay-1,1)]')*RHO_whole;
rho_result = PartialTrace(RHO_whole,N_ResLay+1:N_ResLay+N_StartLay,2*ones(1,N_StartLay+N_ResLay));
end

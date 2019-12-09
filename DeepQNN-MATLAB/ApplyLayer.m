%Application of all unitaries in one given layer. This corresponds to the
%action of the \mathcal{E} on a given start density matrix rho_in.
function rho_out = ApplyLayer(rho_in,U,N_CorrLay,N_PrevLay)

%pho_in (pho_out): input (output) state of the corresponding layer
%N_CorrLay  (N_PrevLay): Number of neurons in the corresponding (previous) Layer 
%U array containing all N_CorrLay Unitaries of the corresponding Layer


%Initialising Input State on all N_CorrLay + N_PrevLay Qubits
RHO_in_whole = kron(rho_in,[1;zeros(2^N_CorrLay -1,1)]*[1;zeros(2^N_CorrLay -1,1)]');

%Calculating Output on all N_CorrLay + N_PrevLay Qubits
RHO_out_whole = RHO_in_whole;
for j= 1:N_CorrLay
    V = Swap(kron(U(:,:,j),eye(2^(N_CorrLay - 1))),[N_PrevLay+1,N_PrevLay+j],2*ones(1,N_CorrLay+N_PrevLay));
    RHO_out_whole = V*RHO_out_whole*V';
end

%Calculating Output state of the Neurons in the corresponding Layer
rho_out = PartialTrace(RHO_out_whole,1:N_PrevLay,2*ones(1,N_CorrLay+N_PrevLay));



end
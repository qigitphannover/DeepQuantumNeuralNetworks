%this script tests RandomUnitary1
N=50;
M=60;
hold on
for l=1:3
    for i=1:3
        x=zeros(1,M);
        for k=1:M     
            for j= 1:N
                x(k)=x(k)+Fidelity(GHZ(l,0), RandomUnitary1(2^l,1.5*k/M,20*i*M/k)*GHZ(l,0));
            end
        x(k)=x(k)/N;
        end
        plot(x)
    end
end
hold off

mode = 'test'; % 'train' or 'test'
example = 1;
id_train = 1;
id_test = 1;

% examples

% GHZ states
%  1: [3,1,3]
%  2: [3,3,3]
%  3: [3,1,3,1,3] training [3,1,3]
%  4: [3,1,3,1,3]
%  5: [4,1,4]
%  6: [4,2,4]
%  7: [4,2,4] sparse
%  8: [4,2,1,2,4]
%  9: [4,2,1,2,4] sparse
% 10: [4,1,4,1,4] training [4,1,4]
% 11: [4,1,4,1,4]
% 12: [5,1,5]
% 13: [5,3,5]
% 14: [5,3,5] sparse
% 15: [5,3,1,3,5]
% 16: [5,3,1,3,5] sparse
% 17: [5,1,5,1,5] training [5,1,5]
% 18: [5,1,5,1,5]
% 19: [3,1,3] denoising GHZ + and -
% 20: [3,1,3] denoising GHZ with random phase
% 21: [3,1,3,1,3] denoising GHZ with random phase, training [3,1,3]
% 22: [3,2,1,3] denoising GHZ with random phase
% 23: [3,2,1,2,3] denoising GHZ with random phase
% 24: [3,2,3] denoising GHZ with random phase
% 25: [3,2,2,1,3,1,3] denoising GHZ with random phase
% 26: [3,2,2,2,3] denoising GHZ with random phase
% 27: [3,4,2,1,3] denoising GHZ with random phase
% 28: [3,1,3] denoising GHZ(phi)
    % train 5: pi/4, 500 pairs with negative phase
    % train 6: pi/4, 500 pairs
    % train 7: pi/4, 1000 pairs
    % train 8: pi/4, 2000 pairs
    % train 9: pi/4, 200 pairs with negative phase
% 29: [4,2,1,2,4], spin-flip and unitary noise

% W-states Dicke(3,1)
% 30: [3,1,3]
% 31: [3,1,3,1,3] training [3,1,3]
% 36: [3,2,1,2,3]

% Dicke states Dicke(4,2)
% 32: [4,1,4]
% 33: [4,1,4,1,4] training [4,1,4]
% 34: [4,2,1,2,4]
% 35: [4,2,1,2,4] sparse

% Cluster state with Gamma = [0,1,0,1; 1,0,1,0; 0,1,0,1; 1,0,1,0]
% 37: [4,1,4]
% 38: [4,1,4,1,4] training [4,1,4]
% 39: [4,2,1,2,4]
% 40: [4,2,1,2,4] sparse

switch mode
    case 'train'
        switch example
            case {1, 3}
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.004;
                momentum = 0.9;
                rmsprop = 0.999;
                iter = 200;
                
                M = [3,1,3];
                ideal = GHZ(3);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end
            case 2
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;                
                iter = 200;
                
                M = [3,3,3];
                ideal = GHZ(3);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end     
            case 4
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;                 
                iter = 200;
                
                M = [3,1,3,1,3];
                ideal = GHZ(3);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end                     
            case {5, 10}
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 1500;
                lambda = 0.00005;
                momentum = 0.93;
                rmsprop = 0.999;
                iter = 100;
                
                M = [4,1,4];
                ideal = GHZ(4);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end
            case 6
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;                
                iter = 200;
                
                M = [4,2,4];
                ideal = GHZ(4);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end 
            case 7
                p1 = 0.0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;
                iter = 200;
                
                M = [4,2,4];
                sparsity = {[1,1,1,0; 0,1,1,1],... 
                    [1,0; 1,1; 1,1; 0,1]};
                ideal = GHZ(4);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter',... 
                    'M', 'sparsity', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M, 'sparse', sparsity);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop, 'sparse', sparsity);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end                
            case 8
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;                 
                iter = 200;
                
                M = [4,2,1,2,4];
                ideal = GHZ(4);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end                 
            case 9
                p1 = 0.0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0033;
                momentum = 0.93;
                rmsprop = 0.999;
                iter = 200;
                
                M = [4,2,1,2,4];
                sparsity = {[1,1,1,0; 0,1,1,1],... 
                    [1,1],...
                    [1; 1],...                    
                    [1,0; 1,1; 1,1; 0,1]};
                ideal = GHZ(4);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter',... 
                    'M', 'sparsity', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M, 'sparse', sparsity);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop, 'sparse', sparsity);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end         
            case 11
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;                 
                iter = 200;
                
                M = [4,1,4,1,4];
                ideal = GHZ(4);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end                
            case {12, 17}
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;
                iter = 200;
                
                M = [5,1,5];
                ideal = GHZ(5);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end
            case 13
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;                
                iter = 200;
                
                M = [5,3,5];
                ideal = GHZ(5);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end 
            case 14
                p1 = 0.0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;
                iter = 200;
                
                M = [5,3,5];
                sparsity = {[1,1,1,0,0; 0,1,1,1,0; 0,0,1,1,1],... 
                    [1,0,0; 1,1,0; 1,1,1; 0,1,1; 0,0,1]};
                ideal = GHZ(5);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter',... 
                    'M', 'sparsity', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M, 'sparse', sparsity);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop, 'sparse', sparsity);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end                
            case 15
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;                 
                iter = 200;
                
                M = [5,3,1,3,5];
                ideal = GHZ(5);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end                 
            case 16
                p1 = 0.0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;
                iter = 200;
                
                M = [5,3,1,3,5];
                sparsity = {[1,1,1,0,0; 0,1,1,1,0; 0,0,1,1,1],... 
                    [1,1,1],...
                    [1; 1; 1],...                    
                    [1,0,0; 1,1,0; 1,1,1; 0,1,1; 0,0,1]};
                ideal = GHZ(5);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter',... 
                    'M', 'sparsity', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M, 'sparse', sparsity);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop, 'sparse', sparsity);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end         
            case 18
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;                 
                iter = 200;
                
                M = [5,1,5,1,5];
                ideal = GHZ(5);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end                                
            case 19
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200; % even
                lambda = 0.0062; 
                momentum = 0.945; 
                rmsprop = 0.999;                  
                iter = 200;
                
                if rem(n1, 2) == 1
                    error('Choose an even number of training pairs n1.')
                end
                nfract = n1/2;
                
                M = [3,1,3];
                ideal1 = GHZ(3);
                ideal2 = GHZ(3, pi);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal1', 'ideal2')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data1 = SpinFlipNoise(ideal1, 2*nfract, p);
                    data2 = SpinFlipNoise(ideal2, 2*nfract, p);
                
                    U1 = QuickInitilizer(M);
                    in = cat(2, data1(:, 1:nfract), data2(:, 1:nfract));
                    train = cat(2, data1(:, (nfract+1):end), data2(:, (nfract+1):end));
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end                
            case {20, 21}
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n_phases = 4;
                phase_max = pi;
                n1 = [100,100,100,100]; % array of length n_phases, number of training pairs for each phase, even
                lambda = 0.00028;
                momentum = 0.98;
                rmsprop = 0.999;                  
                iter = 200;
                
                if any(rem(n1, 2)) == true
                    error('Choose even numbers of training pairs n1.')
                end
                % nfract = n1 / n_phases;
                phases = (0:(n_phases-1)) * phase_max / (n_phases-1);
                
                M = [3,1,3];
                ideal = arrayfun(@(x) GHZ(3, x), phases, 'UniformOutput', false);
                dim = 2^3;
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n_phases', 'phase_max', 'n1',... 
                    'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    in = zeros(dim,sum(n1));
                    train = zeros(dim,sum(n1));
                    for j = 1:n_phases
                        % in(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = ...
                        %      cat(2, SpinFlipNoise(ideal{j}, n1(j)/2, p), SpinFlipNoise(conj(ideal{j}), n1(j)/2, p));
                        in(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = ...
                           cat(2, SpinFlipNoise(ideal{j}, n1(j)/2, p), SpinFlipNoise(ideal{j}, n1(j)/2, p));
                        train(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = SpinFlipNoise(ideal{j}, n1(j), p);
                    end
                
                    U1 = QuickInitilizer(M);
                    % in = data(:, 1:2:end);
                    % train = data(:, 2:2:end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end  
          case 22
                p1 = 0;
                p2 = 0.2;
                pnum = 3;
                n_phases = 5;
                phase_max = pi;
                n1 = [20,20,20,20,20]; % array of length n_phases, number of training pairs for each phase, even
                lambda = 0.005;
                momentum = 0.92;
                rmsprop = 0.999;                  
                iter = 200;
                
                if any(rem(n1, 2)) == true
                    error('Choose even numbers of training pairs n1.')
                end
                % nfract = n1 / n_phases;
                phases = (0:(n_phases-1)) * phase_max / (n_phases-1);
                
                M = [3,2,1,3];
                ideal = arrayfun(@(x) GHZ(3, x), phases, 'UniformOutput', false);
                dim = 2^3;
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n_phases', 'phase_max', 'n1',... 
                    'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    in = zeros(dim,sum(n1));
                    train = zeros(dim,sum(n1));
                    for j = 1:n_phases
                         in(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = ...
                             cat(2, SpinFlipNoise(ideal{j}, n1(j)/2, p), SpinFlipNoise(conj(ideal{j}), n1(j)/2, p));
                        % in(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = ...
                        %    cat(2, SpinFlipNoise(ideal{j}, n1(j)/2, p), SpinFlipNoise(ideal{j}, n1(j)/2, p));
                         train(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = SpinFlipNoise(ideal{j}, n1(j), p);
                    end
                
                    U1 = QuickInitilizer(M);
                    % in = data(:, 1:2:end);
                    % train = data(:, 2:2:end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end
          case 23
                p1 = 0;
                p2 = 0.3;
                pnum = 3;
                n1 = 200; % divisible by n_phases
                lambda = 0.017;
                momentum = 0.83;
                rmsprop = 0.999;                  
                iter = 200;
                n_phases = 8;
                phase_max = pi;
                
                if rem(n1, n_phases) ~= 0
                    error('Choose a number of training pairs n1 divisible by the number of training phases n_phases.')
                end
                nfract = n1 / n_phases;
                phases = (0:(n_phases-1)) * phase_max /n_phases;
                
                M = [3,2,1,2,3];
                ideal = arrayfun(@(x) GHZ(3, x), phases, 'UniformOutput', false);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'n_phases', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = cell2mat(cellfun(@(x) SpinFlipNoise(x, 2*nfract, p), ideal, 'UniformOutput', false));
                
                    U1 = QuickInitilizer(M);
                    in = data(:, 1:2:end);
                    train = data(:, 2:2:end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end
          case 24
                p1 = 0;
                p2 = 0;
                pnum = 1;
                n1 = [50,50,50]; % divisible by n_phases
                lambda = 0.0003;
                momentum = 0.98;
                rmsprop = 0.999;                  
                iter = 200;
                n_phases = 3;
                phase_max = pi;
                
                % if rem(n1, n_phases) ~= 0
                %     error('Choose a number of training pairs n1 divisible by the number of training phases n_phases.')
                % end
                % nfract = n1 / n_phases;
                phases = (0:(n_phases-1)) * phase_max / (n_phases-1);
                
                M = [3,2,3];
                ideal = arrayfun(@(x) GHZ(3, x), phases, 'UniformOutput', false);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'n_phases', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    in = zeros(dim,sum(n1));
                    train = zeros(dim,sum(n1));
                    for j = 1:n_phases
                        in(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = ...
                            cat(2, SpinFlipNoise(ideal{j}, n1(j)/2, p), SpinFlipNoise(conj(ideal{j}), n1(j)/2, p));
                        % in(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = ...
                        %    cat(2, SpinFlipNoise(ideal{j}, n1(j)/2, p), SpinFlipNoise(ideal{j}, n1(j)/2, p));
                        train(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = SpinFlipNoise(ideal{j}, n1(j), p);
                    end
                    
                    U1 = QuickInitilizer(M);
                    % in = data(:, 1:2:end);
                    % train = data(:, 2:2:end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end
          case 25
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200; % divisible by n_phases
                lambda = 0.0167;
                momentum = 0.835;
                rmsprop = 0.999;                  
                iter = 200;
                n_phases = 8;
                phase_max = pi;
                
                if rem(n1, n_phases) ~= 0
                    error('Choose a number of training pairs n1 divisible by the number of training phases n_phases.')
                end
                nfract = n1 / n_phases;
                phases = (0:(n_phases-1)) * phase_max /n_phases;
                
                M = [3,2,2,1,3,1,3];
                ideal = arrayfun(@(x) GHZ(3, x), phases, 'UniformOutput', false);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'n_phases', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = cell2mat(cellfun(@(x) SpinFlipNoise(x, 2*nfract, p), ideal, 'UniformOutput', false));
                
                    U1 = QuickInitilizer(M);
                    in = data(:, 1:2:end);
                    train = data(:, 2:2:end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end
          case 26
                p1 = 0;
                p2 = 0.2;
                pnum = 3;
                n_phases = 5;
                phase_max = pi;
                n1 = [20,20,20,20,20]; % array of length n_phases, number of training pairs for each phase, even
                lambda = 0.005;
                momentum = 0.92;
                rmsprop = 0.999;                  
                iter = 200;
                
                if any(rem(n1, 2)) == true
                    error('Choose even numbers of training pairs n1.')
                end
                % nfract = n1 / n_phases;
                phases = (0:(n_phases-1)) * phase_max / (n_phases-1);
                
                M = [3,2,2,2,3];
                ideal = arrayfun(@(x) GHZ(3, x), phases, 'UniformOutput', false);
                dim = 2^3;
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n_phases', 'phase_max', 'n1',... 
                    'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    in = zeros(dim,sum(n1));
                    train = zeros(dim,sum(n1));
                    for j = 1:n_phases
                         in(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = ...
                             cat(2, SpinFlipNoise(ideal{j}, n1(j)/2, p), SpinFlipNoise(conj(ideal{j}), n1(j)/2, p));
                        % in(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = ...
                        %    cat(2, SpinFlipNoise(ideal{j}, n1(j)/2, p), SpinFlipNoise(ideal{j}, n1(j)/2, p));
                         train(:,(sum(n1(1:(j-1)))+1):sum(n1(1:j))) = SpinFlipNoise(ideal{j}, n1(j), p);
                    end
                
                    U1 = QuickInitilizer(M);
                    % in = data(:, 1:2:end);
                    % train = data(:, 2:2:end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end
          case 27
                p1 = 0;
                p2 = 0.3;
                pnum = 3;
                n1 = 160; % divisible by n_phases
                lambda = 0.0167;
                momentum = 0.835;
                rmsprop = 0.999;                  
                iter = 160;
                n_phases = 8;
                phase_max = pi;
                
                if rem(n1, n_phases) ~= 0
                    error('Choose a number of training pairs n1 divisible by the number of training phases n_phases.')
                end
                nfract = n1 / n_phases;
                phases = (0:(n_phases-1)) * phase_max /n_phases;
                
                M = [3,4,2,1,3];
                ideal = arrayfun(@(x) GHZ(3, x), phases, 'UniformOutput', false);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'n_phases', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = cell2mat(cellfun(@(x) SpinFlipNoise(x, 2*nfract, p), ideal, 'UniformOutput', false));
                
                    U1 = QuickInitilizer(M);
                    in = data(:, 1:2:end);
                    train = data(:, 2:2:end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end
            case 28
                p1 = 0.1;
                p2 = 0.3;
                pnum = 3;
                n1 = 200;
                lambda = 0.00059; % 0.0003 for n1 = 500
                momentum = 0.98; % 0.98 for n1 = 500
                rmsprop = 0.999;
                iter = 200;
                phase = pi/4;
                
                M = [3,1,3];
                ideal = GHZ(3, phase);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'phase', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    % data = SpinFlipNoise(ideal, 2*n1, p);
                    data1 = SpinFlipNoise(ideal, 3/2*n1, p);
                    data2 = SpinFlipNoise(conj(ideal), 1/2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    % in = data(:,1:n1);
                    % train = data(:,(n1+1):end);
                    in = cat(2, data1(:,1:n1/2), data2);
                    train = data1(:,(n1/2+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end
           case 29
                p = 0.2;
                tm = 0.3;
                q = 20;
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;                 
                iter = 200;
                
                M = [4,2,1,2,4];
                ideal = GHZ(4);
                
                data_sf = mat2cell(SpinFlipNoise(ideal, 2*n1, p),[length(ideal)], ones(1,2*n1));
                data = cell2mat(cellfun(@(x) UnitaryNoise(x, 1, tm, q), data_sf, 'UniformOutput', false));
                
                U1 = QuickInitilizer(M);
                in = data(:,1:n1);
                train = data(:,(n1+1):end);
                
                tic
                [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                    'momentum', momentum, 'Nadam', rmsprop);
                tm = toc;
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p', 'tm', 'q', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal',...
                    'in', 'train', 'U1', 'U2', 'CList', 'tm')
            case {30, 31}
                p1 = 0;
                p2 = 0.5;
                pnum = 6;
                n1 = 200;
                lambda = 0.001;
                momentum = 0.97;
                rmsprop = 0.999;
                iter = 150;
                
                M = [3,1,3];
                ideal = Dicke(3,1);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end
            case {32, 33}
                p1 = 0;
                p2 = 0.5;
                pnum = 6;
                n1 = 200;
                lambda = 0.001; % 0.0078
                momentum = 0.95; % 0.84
                rmsprop = 0.999;
                iter = 100;                             
                
                M = [4,1,4];
                ideal = Dicke(4,2);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end                
            case 34
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.001;
                momentum = 0.95;
                rmsprop = 0.999;                 
                iter = 100;
                
                M = [4,2,1,2,4];
                ideal = Dicke(4,2);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end      
            case 35
                p1 = 0.0;
                p2 = 0.5;
                pnum = 6;
                n1 = 200;
                lambda = 0.0033;
                momentum = 0.93;
                rmsprop = 0.999;
                iter = 100;
                
                M = [4,2,1,2,4];
                sparsity = {[1,1,1,0; 0,1,1,1],... 
                    [1,1],...
                    [1; 1],...                    
                    [1,0; 1,1; 1,1; 0,1]};
                ideal = Dicke(4,2);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter',... 
                    'M', 'sparsity', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M, 'sparse', sparsity);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop, 'sparse', sparsity);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end     
            case 36
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0008;
                momentum = 0.98;
                rmsprop = 0.999;
                iter = 200;
                
                M = [3,2,1,2,3];
                ideal = Dicke(3,1);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end     
            case {37, 38}
                p1 = 0;
                p2 = 0.4;
                pnum = 5;
                n1 = 200;
                lambda = 0.005; % 0.0078
                momentum = 0.9; % 0.84
                rmsprop = 0.999;
                iter = 150;                             
                
                M = [4,1,4];
                Gamma = [0,1,0,1; 1,0,1,0; 0,1,0,1; 1,0,1,0];
                ideal = Cluster(Gamma);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end                
            case 39
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200;
                lambda = 0.0025;
                momentum = 0.93;
                rmsprop = 0.999;                 
                iter = 150;
                
                M = [4,2,1,2,4];
                Gamma = [0,1,0,1; 1,0,1,0; 0,1,0,1; 1,0,1,0];
                ideal = Cluster(Gamma);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end      
            case 40
                p1 = 0.0;
                p2 = 0.5;
                pnum = 6;
                n1 = 200;
                lambda = 0.0033;
                momentum = 0.93;
                rmsprop = 0.999;
                iter = 100;
                
                M = [4,2,1,2,4];
                sparsity = {[1,1,1,0; 0,1,1,1],... 
                    [1,1],...
                    [1; 1],...                    
                    [1,0; 1,1; 1,1; 0,1]};
                Gamma = [0,1,0,1; 1,0,1,0; 0,1,0,1; 1,0,1,0];
                ideal = Cluster(Gamma);
                
                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter',... 
                    'M', 'sparsity', 'ideal')
                
                if pnum == 1
                    dp = 0;
                else
                    dp = (p2-p1)/(pnum-1);
                end
                for i = 1:pnum
                    p = p1 + (i-1)*dp;
                    fprintf('\n p = %1.3f \n\n', p)
                    data = SpinFlipNoise(ideal, 2*n1, p);
                
                    U1 = QuickInitilizer(M, 'sparse', sparsity);
                    in = data(:,1:n1);
                    train = data(:,(n1+1):end);
                
                    tic
                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                        'momentum', momentum, 'Nadam', rmsprop, 'sparse', sparsity);
                    tm = toc;
                
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                end     
            otherwise
                error('Choose an example from 1 to 40.')               
        end
        
        
    case 'test'
        switch example
            case {1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 28, 30, 32, 34, 35, 36, 37, 39, 40}
                % n2 = 1500;
             
                load(strcat('Dat/dsf_train_ex', num2str(example), '_train',  num2str(id_train), '.mat'),...
                    'pnum', 'n1', 'M', 'ideal')     
                n2 = n1;
                dim = length(ideal);
                
                save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test), '.mat'),...
                    'n2', 'pnum', 'M', 'ideal')
                
                for i = 1:pnum
                    load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'U2')
                    
                    in = SpinFlipNoise(ideal, n2, p);
                    % load(strcat('Dat/dsf_test_ex10_train6_test1_p', num2str(i), '.mat'), 'in')
                    target = repmat(ideal, 1, n2);
                    
                    out = zeros(dim,dim,n2);
                    for j = 1:n2
                        out(:,:,j) = ApplyNetwork(in(:,j), U2, M);
                    end
                    
                    fid_in = Fidelity(target, in);
                    fid_out = Fidelity(target, out);
                    fid_ideal = Fidelity(ideal, ApplyNetwork(ideal, U2, M));
                    meanfid_in = mean(fid_in);
                    meanfid_out = mean(fid_out);
                    varfid_in = var(fid_in);
                    varfid_out = var(fid_out);
                    
                    save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                        num2str(id_train), '_test', num2str(id_test), '_p', num2str(i), '.mat'),...
                        'p', 'U2', 'in', 'out', 'target',...
                        'fid_in', 'fid_out', 'fid_ideal', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out')
                end           
            case {3, 10, 17, 31, 33, 38}
                % n2 = 200;
                if example == 31
                    load(strcat('Dat/dsf_train_ex', num2str(30), '_train',  num2str(id_train), '.mat'),...
                    'pnum', 'n1', 'ideal')     
                else
                    load(strcat('Dat/dsf_train_ex', num2str(example), '_train',  num2str(id_train), '.mat'),...
                    'pnum', 'n1', 'ideal') 
                end
                n2 = n1;
                dim = length(ideal);
                m = log2(dim);
                M = [m, 1, m, 1, m];
                
                save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test), '.mat'),...
                    'n2', 'pnum', 'M', 'ideal')
                
                for i = 1:pnum
                    if example == 31
                        load(strcat('Dat/dsf_train_ex', num2str(30), '_train',...
                            num2str(id_train), '_p', num2str(i), '.mat'),...
                            'p', 'U2')
                    else
                        load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                            num2str(id_train), '_p', num2str(i), '.mat'),...
                            'p', 'U2')                        
                    end
                    U2 = [U2, U2(2:end)];
                    
                    % load(strcat('Dat/dsf_test_ex36_train4_test1_p', num2str(i), '.mat'), 'in')
                    in = SpinFlipNoise(ideal, n2, p);
                    target = repmat(ideal, 1, n2);
                    
                    out = zeros(dim,dim,n2);
                    for j = 1:n2
                        out(:,:,j) = ApplyNetwork(in(:,j), U2, M);
                    end
                    
                    fid_in = Fidelity(target, in);
                    fid_out = Fidelity(target, out);
                    fid_ideal = Fidelity(ideal, ApplyNetwork(ideal, U2, M));
                    meanfid_in = mean(fid_in);
                    meanfid_out = mean(fid_out);
                    varfid_in = var(fid_in);
                    varfid_out = var(fid_out);
                    
                    save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                        num2str(id_train), '_test', num2str(id_test), '_p', num2str(i), '.mat'),...
                        'p', 'U2', 'in', 'out', 'target',...
                        'fid_in', 'fid_out', 'fid_ideal', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out')
                end                           
            case 19
                % n2 = 200;
             
                load(strcat('Dat/dsf_train_ex', num2str(example), '_train',  num2str(id_train), '.mat'),...
                    'pnum', 'n1', 'M', 'ideal1', 'ideal2')     
                n2 = n1;
                dim = length(ideal1);
                
                if rem(n2, 2) == 1
                    error('Choose an even number of testing pairs n2.')
                end
                nfract = n2/2;                
                
                save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test), '.mat'),...
                    'n2', 'pnum', 'M', 'ideal1', 'ideal2')
                
                for i = 1:pnum
                    load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'U2')
                    
                    in = cat(2, SpinFlipNoise(ideal1, nfract, p), SpinFlipNoise(ideal2, nfract, p));
                    target = cat(2, repmat(ideal1, 1, nfract), repmat(ideal2, 1, nfract));
                    
                    out = zeros(dim,dim,n2);
                    for j = 1:n2
                        out(:,:,j) = ApplyNetwork(in(:,j), U2, M);
                    end
                    
                    fid_in = Fidelity(target, in);
                    fid_out = Fidelity(target, out);
                    % fid_ideal = Fidelity(ideal, ApplyNetwork(ideal, U2, M));
                    meanfid_in = mean(fid_in);
                    meanfid_out = mean(fid_out);
                    varfid_in = var(fid_in);
                    varfid_out = var(fid_out);
                    
                    save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                        num2str(id_train), '_test', num2str(id_test), '_p', num2str(i), '.mat'),...
                        'p', 'U2', 'in', 'out', 'target',...
                        'fid_in', 'fid_out', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out')
                end                           
            case {20, 22, 23, 24, 25, 26, 27}
                n2 = 200;
             
                load(strcat('Dat/dsf_train_ex', num2str(example), '_train',  num2str(id_train), '.mat'),...
                    'pnum', 'n1', 'n_phases', 'M', 'ideal')     
                % n2 = n1;
                dim = length(ideal{1});
                m = log2(dim);
                phase_max = pi;
                
                phases = phase_max * rand(1,n2);
                % phases = repmat((0:(n_phases-1)) * phase_max / (n_phases-1),1,n2/n_phases);
                % phases = ones(1,n2) * pi;
                
                save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test), '.mat'),...
                    'n2', 'pnum', 'n_phases', 'M', 'ideal')
                
                for i = 1:pnum
                    load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'U2')
                    
                    target = GHZ(m, phases);
                    target_cell = num2cell(target, 1);
                    [in_cell, filter] = cellfun(@(x) SpinFlipNoise(x, 1, p), target_cell, 'UniformOutput', false);
                    in = cell2mat(in_cell);
                    filter = cell2mat(filter);
                    % in1 = target;
                    % in2 = zeros(dim,3*n2);
                    % flip = [0,1;1,0];
                    % flip1 = kron(kron(flip,eye(2)),eye(2));
                    % flip2 = kron(kron(eye(2),flip),eye(2));
                    % flip3 = kron(kron(eye(2),eye(2)),flip);
                    % for k = 1:n2
                    %     targetk = target(:,k);
                    %     in2(:,3*k-2) = flip1 * targetk;
                    %     in2(:,3*k-1) = flip2 * targetk;
                    %     in2(:,3*k) = flip3 * targetk;
                    % end
                    
                    % out1 = zeros(dim,dim,n2);
                    % out2 = zeros(dim,dim,3*n2);
                    for j = 1:n2
                        out(:,:,j) = ApplyNetwork(in(:,j), U2, M);
                    end
                    out_filtered = out(:,:,filter);
                    % for j = 1:3*n2
                    %     out2(:,:,j) = ApplyNetwork(in2(:,j), U2, M);
                    % end
                    
                    fid_in = Fidelity(target, in);
                    meanfid_in = mean(fid_in);
                    varfid_in = var(fid_in);
                    fid_inf = Fidelity(target(:,filter), in(:,filter));
                    meanfid_inf = mean(fid_inf);
                    varfid_inf = var(fid_inf);                    
                    fid_out = Fidelity(target, out);
                    meanfid_out = mean(fid_out);
                    varfid_out = var(fid_out);
                    fid_outf = Fidelity(target(:,filter), out_filtered);
                    meanfid_outf = mean(fid_outf);
                    varfid_outf = var(fid_outf);                    
                    
                    save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                        num2str(id_train), '_test', num2str(id_test), '_p', num2str(i), '.mat'),...
                        'p', 'U2', 'in', 'out', 'target',...
                        'fid_in', 'fid_inf', 'meanfid_in', 'meanfid_inf', 'varfid_in', 'varfid_inf',...
                        'fid_out', 'fid_outf', 'meanfid_out', 'meanfid_outf', 'varfid_out', 'varfid_outf')
                end                 
            case 21
                % n2 = 200;
             
                load(strcat('Dat/dsf_train_ex', num2str(example), '_train',  num2str(id_train), '.mat'),...
                    'pnum', 'n1', 'n_phases', 'ideal')     
                n2 = n1;
                dim = length(ideal{1});
                m = log2(dim);
                M = [m, 1, m, 1, m];
                phase_max = 1;
                
                if rem(n1, n_phases) ~= 0
                    error('Choose a number of training pairs n1 divisible by the number of training phases n_phases.')
                end
                phases = phase_max*rand(1,n2);
                
                save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test), '.mat'),...
                    'n2', 'pnum', 'n_phases', 'M', 'ideal')
                
                for i = 1:pnum
                    load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'U2')
                    U2 = [U2, U2(2:end)];
                    
                    target = GHZ(m, phases);
                    target_cell = num2cell(target, 1);
                    in_cell = cellfun(@(x) SpinFlipNoise(x, 1, p), target_cell, 'UniformOutput', false);
                    in = cell2mat(in_cell);
                    
                    out = zeros(dim,dim,n2);
                    for j = 1:n2
                        out(:,:,j) = ApplyNetwork(in(:,j), U2, M);
                    end
                    
                    fid_in = Fidelity(target, in);
                    fid_out = Fidelity(target, out);
                    meanfid_in = mean(fid_in);
                    meanfid_out = mean(fid_out);
                    varfid_in = var(fid_in);
                    varfid_out = var(fid_out);
                    
                    save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                        num2str(id_train), '_test', num2str(id_test), '_p', num2str(i), '.mat'),...
                        'p', 'U2', 'in', 'out', 'target',...
                        'fid_in', 'fid_out', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out')
                end       
            case 29
                % n2 = 200;
             
                load(strcat('Dat/dsf_train_ex', num2str(example), '_train',  num2str(id_train), '.mat'),...
                    'n1', 'M', 'ideal', 'p', 'tm', 'q', 'U2')     
                n2 = n1;
                dim = length(ideal);
                m = log2(dim);
   
                target = repmat(ideal, 1, n2);
                in_sf = mat2cell(SpinFlipNoise(ideal, n2, p), dim, ones(1,n2));
                in = cell2mat(cellfun(@(x) UnitaryNoise(x, 1, tm, q), in_sf, 'UniformOutput', false));
                
                out = zeros(dim,dim,n2);
                for j = 1:n2
                    out(:,:,j) = ApplyNetwork(in(:,j), U2, M);
                end
                
                fid_in = Fidelity(target, in);
                fid_out = Fidelity(target, out);
                meanfid_in = mean(fid_in);
                meanfid_out = mean(fid_out);
                varfid_in = var(fid_in);
                varfid_out = var(fid_out);
                
                save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test), '.mat'),...
                    'n2', 'M', 'ideal', 'p', 'tm', 'q', 'U2',...
                    'in', 'out', 'target',...
                    'fid_in', 'fid_out', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out')
            otherwise
                error('Choose an example from 1 to 40.')
        end        
    otherwise
        error('Valid modes are "train" and "test".')        
end

mode = 'train'; % 'train' or 'test'
example = 1;
id_train = 8;
id_test = 8;
% examples
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

switch mode
    case 'train'
        switch example
            case {1, 3}
                p1 = 0.0;
                p2 = 0.0;
                pnum = 1;
                n1 = 100;
                lambda = 0.0040;
                %lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;
                %set different lambda
                if_variable_lambda = true;
                lambda_list = [0.0030,0.0035,0.0040,0.0045,0.0050];
                %set start and end time t, if if_variable_n is true and number of different time evaluation points
                if_variable_t = true;
                t_start = 0.0500;
                t_end = 0.0500;
                t_number = 1;
                n_default = 20;
                %set start and end time n, if if_variable_n is true and number of different time evaluation points
                if_variable_n = false;
                n_start = 10;
                n_end = 40;
                n_number = 4;
                t_default = 0.005;

                iter = 100;
                %iter = 3;

                M = [3,1,3];
                ideal = GHZ(3);

                save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                        'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal','t_number','n_number','if_variable_t','if_variable_n','if_variable_lambda')


                if if_variable_lambda
                    for lambda_count = 1:length(lambda_list)
                        fprintf('lambda = %1.5f \n\n', lambda_list(lambda_count));
                        lambda = lambda_list(lambda_count);

                        %variable t and n is not allowed (need to be implemented)
                        if( if_variable_t & if_variable_n )
                            error('Variable_t and variable_n is not allowed. Please choose just one variable parameter of these. Otherwise implement it.')
                        end

                        if pnum == 1
                            p_step = 0;
                        else
                            p_step = (p2-p1)/(pnum-1);
                        end
                        for p_count = 1:pnum
                            p = p1 + (p_count-1)*p_step;
                            fprintf('p = %1.3f \n\n', p)
                            data = SpinFlipNoise(ideal, 2*n1, p);

                            U1 = QuickInitilizer(M);
                            in = data(:,1:n1);
                            train = data(:,(n1+1):end);

                            tic

                            %If there is variable_t
                            if (if_variable_t)
                                if t_number == 1
                                    t_step = 0;
                                else
                                    %calculate time step for interval
                                    t_step = (t_end - t_start)/(t_number-1);
                                end

                                %Add noise to ML on different time points (time dependend noise)
                                for t_count = 1:t_number
                                    t = t_start + (t_count - 1)*t_step;
                                    %fprintf('\n t = %1.3f \n\n', t)
                                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                                        'momentum', momentum, 'Nadam', rmsprop, 'variable_t', t, 'n_default', n_default);
                                    tm = toc;
                                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                                        num2str(id_train), '_p', num2str(p_count), '_t', num2str(t_count), '_lambda', num2str(lambda_count), '.mat'),...
                                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm','t','n_default','lambda')
                                end
                            %If there is variable_n
                            elseif (if_variable_n)
                                if n_number == 1
                                    n_step = 0;
                                else
                                    %calculate time step for interval
                                    n_step = (n_end - n_start)/(n_number-1);
                                end

                                %Add noise to ML on different time points (time dependend noise)
                                for n_count = 1:n_number
                                    n = n_start + (n_count - 1)*n_step;
                                    %fprintf('\n n = %1.3f \n\n', n)
                                    [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                                        'momentum', momentum, 'Nadam', rmsprop, 'variable_n', n, 't_default', t_default);
                                    tm = toc;
                                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                                        num2str(id_train), '_p', num2str(p_count), '_n', num2str(n_count), '_lambda', num2str(lambda_count), '.mat'),...
                                        'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm','t_default','n','lambda')
                                end
                             else
                                [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                                    'momentum', momentum, 'Nadam', rmsprop);
                                tm = toc;
                                save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                                    num2str(id_train), '_p', num2str(p_count), '.mat'),...
                                    'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                            end

                        %p for end
                        end


                    %lamba for end
                    end

                else
                    save(strcat('Dat/dsf_train_ex', num2str(example), '_train', num2str(id_train), '.mat'),...
                    'p1', 'p2', 'pnum', 'n1', 'lambda', 'momentum', 'rmsprop', 'iter', 'M', 'ideal','t_number','n_number','if_variable_t','if_variable_n')

                    %variable t and n is not allowed (need to be implemented)
                    if( if_variable_t & if_variable_n )
                        error('Variable_t and variable_n is not allowed. Please choose just one variable parameter of these. Otherwise implement it.')
                    end

                    if pnum == 1
                        p_step = 0;
                    else
                        p_step = (p2-p1)/(pnum-1);
                    end
                    for p_count = 1:pnum
                        p = p1 + (p_count-1)*p_step;
                        fprintf('p = %1.3f \n\n', p)
                        data = SpinFlipNoise(ideal, 2*n1, p);

                        U1 = QuickInitilizer(M);
                        in = data(:,1:n1);
                        train = data(:,(n1+1):end);

                        tic

                        %If there is variable_t
                        if (if_variable_t)
                            if t_number == 1
                                t_step = 0;
                            else
                                %calculate time step for interval
                                t_step = (t_end - t_start)/(t_number-1);
                            end

                            %Add noise to ML on different time points (time dependend noise)
                            for t_count = 1:t_number
                                t = t_start + (t_count - 1)*t_step;
                                %fprintf('\n t = %1.3f \n\n', t)
                                [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                                    'momentum', momentum, 'Nadam', rmsprop, 'variable_t', t, 'n_default', n_default);
                                tm = toc;
                                save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                                    num2str(id_train), '_p', num2str(p_count), '_t', num2str(t_count), '.mat'),...
                                    'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm','t','n_default')
                            end
                        %If there is variable_n
                        elseif (if_variable_n)
                            if n_number == 1
                                n_step = 0;
                            else
                                %calculate time step for interval
                                n_step = (n_end - n_start)/(n_number-1);
                            end

                            %Add noise to ML on different time points (time dependend noise)
                            for n_count = 1:n_number
                                n = n_start + (n_count - 1)*n_step;
                                %fprintf('\n n = %1.3f \n\n', n)
                                [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                                    'momentum', momentum, 'Nadam', rmsprop, 'variable_n', n, 't_default', t_default);
                                tm = toc;
                                save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                                    num2str(id_train), '_p', num2str(p_count), '_n', num2str(n_count), '.mat'),...
                                    'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm','t_default','n')
                            end
                         else
                            [U2, CList] = TrainNetwork(in, train, U1, M, lambda, iter,...
                                'momentum', momentum, 'Nadam', rmsprop);
                            tm = toc;
                            save(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                                num2str(id_train), '_p', num2str(p_count), '.mat'),...
                                'p', 'in', 'train', 'U1', 'U2', 'CList', 'tm')
                        end

                    %for p ends
                    end

                %if_lambda ends
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
                n1 = 200;
                lambda = 0.0175;
                momentum = 0.8;
                rmsprop = 0.999;
                iter = 200;

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
                lambda = 0.0175;
                momentum = 0.8;
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
                lambda = 0.0175;
                momentum = 0.8;
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
            case 20
                p1 = 0;
                p2 = 0.5;
                pnum = 11;
                n1 = 200; % divisible by n_phases
                lambda = 0.0172;
                momentum = 0.82;
                rmsprop = 0.999;
                iter = 200;
                n_phases = 8;

                if rem(n1, n_phases) ~= 0
                    error('Choose a number of training pairs n1 divisible by the number of training phases n_phases.')
                end
                nfract = n1 / n_phases;
                phases = (0:(n_phases-1)) * 2*pi/n_phases;

                M = [3,1,3];
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
            otherwise
                error('Choose an example from 1 to 20.')
        end


    case 'test'
        switch example
            case {1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18}
                % n2 = 200;

                %%IMPORTANT: Use for test if to run old trained QNNs (if not, it will NOT run, or with old chached values!)
                %if_variable_lambda = false;
                %if_variable_t = true;
                %if_variable_n = false;


                load(strcat('Dat/dsf_train_ex', num2str(example), '_train',  num2str(id_train), '.mat'),...
                    'pnum', 'n1', 'M', 'ideal','t_number','n_number','if_variable_t','if_variable_n')
                n2 = n1;
                dim = length(ideal);

                save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test), '.mat'),...
                    'n2', 'pnum', 'M', 'ideal','t_number','n_number')

                %variable t and n is not allowed (need to be implemented)
                if( if_variable_t & if_variable_n )
                    error('Variable_t and variable_n is not allowed. Please choose just one variable parameter of these. Otherwise implement it.')
                end

                if (if_variable_t)

                    for p_count = 1:pnum
                        load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                            num2str(id_train), '_p', num2str(p_count), '_t', num2str(t_count), '.mat'),...
                            'p', 'U2','t','n_default')

                        %Just one in and target for all t
                        in = SpinFlipNoise(ideal, n2, p);
                        target = repmat(ideal, 1, n2);

                        for t_count = 1:t_number
                            if if_variable_lambda
                                for lambda_count = 1:length(lambda_list)

                                    load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                                        num2str(id_train), '_p', num2str(p_count), '_t', num2str(t_count), '_lambda', num2str(lambda_count), '.mat'),...
                                        'p', 'U2','t','n_default','lambda')

                                    %Old version:new in and target for every t
                                    %in = SpinFlipNoise(ideal, n2, p);
                                    %target = repmat(ideal, 1, n2);

                                    out = zeros(dim,dim,n2);
                                    for j = 1:n2
                                        out(:,:,j) = ApplyNetwork(in(:,j), U2, M, 'variable_t', t, 'n_default', n_default);
                                    end

                                    fid_in = Fidelity(target, in);
                                    fid_out = Fidelity(target, out);
                                    fid_ideal = Fidelity(ideal, ApplyNetwork(ideal, U2, M, 'variable_t', t, 'n_default', n_default));
                                    meanfid_in = mean(fid_in);
                                    meanfid_out = mean(fid_out);
                                    varfid_in = var(fid_in);
                                    varfid_out = var(fid_out);

                                    save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                                        num2str(id_train), '_test', num2str(id_test), '_p', num2str(p_count), '_t', num2str(t_count), '_lambda', num2str(lambda_count), '.mat'),...
                                        'p', 'U2', 'in', 'out', 'target',...
                                        'fid_in', 'fid_out', 'fid_ideal', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out','t','n_default','lambda')

                                end

                            else

                                load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                                    num2str(id_train), '_p', num2str(p_count), '_t', num2str(t_count), '.mat'),...
                                    'p', 'U2','t','n_default')

                                %Old version:new in and target for every t
                                %in = SpinFlipNoise(ideal, n2, p);
                                %target = repmat(ideal, 1, n2);

                                out = zeros(dim,dim,n2);
                                for j = 1:n2
                                    out(:,:,j) = ApplyNetwork(in(:,j), U2, M, 'variable_t', t, 'n_default', n_default);
                                end

                                fid_in = Fidelity(target, in);
                                fid_out = Fidelity(target, out);
                                fid_ideal = Fidelity(ideal, ApplyNetwork(ideal, U2, M, 'variable_t', t, 'n_default', n_default));
                                meanfid_in = mean(fid_in);
                                meanfid_out = mean(fid_out);
                                varfid_in = var(fid_in);
                                varfid_out = var(fid_out);

                                save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                                    num2str(id_train), '_test', num2str(id_test), '_p', num2str(p_count), '_t', num2str(t_count), '.mat'),...
                                    'p', 'U2', 'in', 'out', 'target',...
                                    'fid_in', 'fid_out', 'fid_ideal', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out','t','n_default')
                            end

                        end

                    end

                elseif (if_variable_n)

                    for p_count = 1:pnum
                        load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                            num2str(id_train), '_p', num2str(p_count), '_n', num2str(n_count), '.mat'),...
                            'p', 'U2','n','t_default')

                        %Just one in and target for all n
                        in = SpinFlipNoise(ideal, n2, p);
                        target = repmat(ideal, 1, n2);

                        for n_count = 1:n_number
                            load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                                num2str(id_train), '_p', num2str(p_count), '_n', num2str(n_count), '.mat'),...
                                'p', 'U2','n','t_default')

                            %Old version:new in and target for every n
                            %in = SpinFlipNoise(ideal, n2, p);
                            %target = repmat(ideal, 1, n2);

                            out = zeros(dim,dim,n2);
                            for j = 1:n2
                                out(:,:,j) = ApplyNetwork(in(:,j), U2, M, 'variable_n', n, 't_default', t_default);
                            end

                            fid_in = Fidelity(target, in);
                            fid_out = Fidelity(target, out);
                            fid_ideal = Fidelity(ideal, ApplyNetwork(ideal, U2, M, 'variable_n', n, 't_default', t_default));
                            meanfid_in = mean(fid_in);
                            meanfid_out = mean(fid_out);
                            varfid_in = var(fid_in);
                            varfid_out = var(fid_out);

                            save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                                num2str(id_train), '_test', num2str(id_test), '_p', num2str(p_count), '_n', num2str(n_count), '.mat'),...
                                'p', 'U2', 'in', 'out', 'target',...
                                'fid_in', 'fid_out', 'fid_ideal', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out','n','t_default')
                        end

                    end

                else

                    for p_count = 1:pnum
                        load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                            num2str(id_train), '_p', num2str(p_count), '.mat'),...
                            'p', 'U2')

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
                            num2str(id_train), '_test', num2str(id_test), '_p', num2str(p_count), '.mat'),...
                            'p', 'U2', 'in', 'out', 'target',...
                            'fid_in', 'fid_out', 'fid_ideal', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out')
                    end

                end

            case {3, 10, 17}
                % n2 = 200;

                %%IMPORTANT: Use for test if to run old trained QNNs (if not, it will NOT run, or with old chached values!)
                %if_variable_lambda = false;
                %if_variable_t = true;
                %if_variable_n = false;

                load(strcat('Dat/dsf_train_ex', num2str(example), '_train',  num2str(id_train), '.mat'),...
                    'pnum', 'n1', 'ideal','t_number','n_number')
                n2 = n1;
                dim = length(ideal);
                m = log2(dim);
                M = [m, 1, m, 1, m];

                save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test), '.mat'),...
                    'n2', 'pnum', 'M', 'ideal')

                %variable t and n is not allowed (need to be implemented)
                if( if_variable_t & if_variable_n )
                    error('Variable_t and variable_n is not allowed. Please choose just one variable parameter of these. Otherwise implement it.')
                end

                if (if_variable_t)
                    for p_count = 1:pnum
                        load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                            num2str(id_train), '_p', num2str(p_count), '_t', num2str(t_count), '.mat'),...
                            'p', 'U2','t','n_default')


                        %Just one in and target for all t
                        in = SpinFlipNoise(ideal, n2, p);
                        target = repmat(ideal, 1, n2);

                        for t_count = 1:t_number
                            load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                                num2str(id_train), '_p', num2str(p_count), '_t', num2str(t_count), '.mat'),...
                                'p', 'U2','t','n_default')

                            U2 = [U2, U2(2:end)]; %#ok<AGROW>

                            %Old version:new in and target for every t
                            %in = SpinFlipNoise(ideal, n2, p);
                            %target = repmat(ideal, 1, n2);

                            out = zeros(dim,dim,n2);
                            for j = 1:n2
                                out(:,:,j) = ApplyNetwork(in(:,j), U2, M, 'variable_t', t, 'n_default', n_default);
                            end

                            fid_in = Fidelity(target, in);
                            fid_out = Fidelity(target, out);
                            fid_ideal = Fidelity(ideal, ApplyNetwork(ideal, U2, M, 'variable_t', t, 'n_default', n_default));
                            meanfid_in = mean(fid_in);
                            meanfid_out = mean(fid_out);
                            varfid_in = var(fid_in);
                            varfid_out = var(fid_out);

                            save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                                num2str(id_train), '_test', num2str(id_test), '_p', num2str(p_count), '_t', num2str(t_count), '.mat'),...
                                'p', 'U2', 'in', 'out', 'target',...
                                'fid_in', 'fid_out', 'fid_ideal', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out','t','n_default')
                        end

                    end

                elseif (if_variable_n)

                    for p_count = 1:pnum
                        load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                            num2str(id_train), '_p', num2str(p_count), '_n', num2str(n_count), '.mat'),...
                            'p', 'U2','n','t_default')

                            %Just one in and target for all n
                            in = SpinFlipNoise(ideal, n2, p);
                            target = repmat(ideal, 1, n2);

                        for n_count = 1:n_number
                            load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                                num2str(id_train), '_p', num2str(p_count), '_n', num2str(n_count), '.mat'),...
                                'p', 'U2','n','t_default')

                            U2 = [U2, U2(2:end)]; %#ok<AGROW>

                            %Old version:new in and target for every n
                            %in = SpinFlipNoise(ideal, n2, p);
                            %target = repmat(ideal, 1, n2);

                            out = zeros(dim,dim,n2);
                            for j = 1:n2
                                out(:,:,j) = ApplyNetwork(in(:,j), U2, M, 'variable_n', n, 't_default', t_default);
                            end

                            fid_in = Fidelity(target, in);
                            fid_out = Fidelity(target, out);
                            fid_ideal = Fidelity(ideal, ApplyNetwork(ideal, U2, M, 'variable_n', n, 't_default', t_default));
                            meanfid_in = mean(fid_in);
                            meanfid_out = mean(fid_out);
                            varfid_in = var(fid_in);
                            varfid_out = var(fid_out);

                            save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                                num2str(id_train), '_test', num2str(id_test), '_p', num2str(p_count), '_n', num2str(n_count), '.mat'),...
                                'p', 'U2', 'in', 'out', 'target',...
                                'fid_in', 'fid_out', 'fid_ideal', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out','n','t_default')
                        end

                    end

                else

                    for p_count = 1:pnum
                        load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                            num2str(id_train), '_p', num2str(p_count), '.mat'),...
                            'p', 'U2')

                            U2 = [U2, U2(2:end)]; %#ok<AGROW>

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
                                num2str(id_train), '_test', num2str(id_test), '_p', num2str(p_count), '.mat'),...
                                'p', 'U2', 'in', 'out', 'target',...
                                'fid_in', 'fid_out', 'fid_ideal', 'meanfid_in', 'meanfid_out', 'varfid_in', 'varfid_out')

                    end

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
            case 20
                % n2 = 200;

                load(strcat('Dat/dsf_train_ex', num2str(example), '_train',  num2str(id_train), '.mat'),...
                    'pnum', 'n1', 'n_phases', 'M', 'ideal')
                n2 = n1;
                dim = length(ideal{1});
                m = log2(dim);

                if rem(n2, 2) == 1
                    error('Choose an even number of testing pairs n2.')
                end
                phases = rand(1,n2);

                save(strcat('Dat/dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test), '.mat'),...
                    'n2', 'pnum', 'n_phases', 'M', 'ideal')

                for i = 1:pnum
                    load(strcat('Dat/dsf_train_ex', num2str(example), '_train',...
                        num2str(id_train), '_p', num2str(i), '.mat'),...
                        'p', 'U2')

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
            otherwise
                error('Choose an example from 1 to 20.')
        end
    otherwise
        error('Valid modes are "train" and "test".')
end

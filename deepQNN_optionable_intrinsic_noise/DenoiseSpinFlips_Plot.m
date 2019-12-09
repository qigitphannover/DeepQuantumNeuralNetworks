
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

%%%%IMPORTANT just use this for old trained data, else comment out
%if_variable_t = true;
%if_variable_n = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch mode
    case 'train'
        switch example
            case {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

                savename = strcat('dsf_train_ex', num2str(example), '_train', num2str(id_train));

                    % load from example_train (title)
                    load(strcat('Dat/', savename, '.mat'), 'pnum', 't_number', 'n_number','if_variable_t','if_variable_n','if_variable_lambda')
                    % title = strcat('n=', num2str(n1), ', l=', num2str(lambda));

                    %plot cost vs iteration with different t
                    if (if_variable_t)
                        % load from example_train_p, legend
                        cost_t = cell(1,t_number);
                        legend_t = cell(1, t_number);

                        for p_fixed_count = 1:pnum

                            for t_count = 1:t_number

                                if if_variable_lambda
                                    cost_lambda = cell(1,length(lambda_list));
                                    legend_lambda = cell(1,length(lambda_list));

                                    for lambda_count = 1:length(lambda_list)
                                        load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_t', num2str(t_count), '_lambda', num2str(lambda_count), '.mat'), 'p', 'CList','t', 'lambda', 'n_default')

                                        cost_lambda{lambda_count} = CList;
                                        legend_lambda{lambda_count} = strcat('lambda =', num2str(lambda));
                                    end

                                    % plot
                                    title_lambda = strcat('train: n=', num2str(n_default), ' p=', num2str(p), ' t=', num2str(t));
                                    PlotCost(cost_lambda, 'savename', savename, 'legend', legend_lambda, 'title', title_lambda)
                                    % save
                                    % save(strcat('Dat/plot_', savename), 'cost', 'legend', 'title')
                                    save(strcat('Dat/plot_', savename, '_t', num2str(t_count), '_p', num2str(p_fixed_count)), 'cost_lambda', 'legend_lambda','title_lambda')

                                else

                                    load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_t', num2str(t_count), '.mat'), 'p', 'CList','t', 'n_default')
                                    cost_t{t_count} = CList;
                                    legend_t{t_count} = strcat('t=', num2str(t));
                                end

                            %end t for
                            end

                            %this needs to be outside t for, so can't be written in the else branch of if_variable_lambda before
                            if ~(if_variable_lambda)
                                % plot
                                title_t = strcat('train: n=', num2str(n_default), ' p=', num2str(p));
                                PlotCost(cost_t, 'savename', savename, 'legend', legend_t, 'title', title_t)
                                % save
                                % save(strcat('Dat/plot_', savename), 'cost', 'legend', 'title')
                                save(strcat('Dat/plot_', savename, '_t', num2str(t_count)), 'cost_t', 'legend_t','title_t')
                            end

                        %end p for
                        end

                    elseif (if_variable_n)
                        % load from example_train_p, legend
                        cost_n = cell(1,n_number);
                        legend_n = cell(1, n_number);

                        for p_fixed_count = 1:pnum

                            for n_count = 1:n_number

                                if if_variable_lambda

                                    for lambda_count = 1:length(lambda_list)
                                        load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_n', num2str(n_count), '_lambda', num2str(lambda_count), '.mat'), 'p', 'CList','n', 'lambda', 't_default')

                                        cost_lambda{lambda_count} = CList;
                                        legend_lambda{lambda_count} = strcat('lambda =', num2str(lambda));
                                    end

                                    % plot
                                    title_lambda = strcat('train: n=', num2str(n), ' p=', num2str(p), ' t=', num2str(t_default));
                                    PlotCost(cost_lambda, 'savename', savename, 'legend', legend_lambda, 'title', title_lambda)
                                    % save
                                    % save(strcat('Dat/plot_', savename), 'cost', 'legend', 'title')
                                    save(strcat('Dat/plot_', savename, '_n', num2str(n_count), '_p', num2str(p_fixed_count)), 'cost_lambda', 'legend_lambda','title_lambda')

                                else

                                    load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_n', num2str(n_count), '.mat'), 'p', 'CList','n','t_default')
                                    cost_n{n_count} = CList;
                                    legend_n{n_count} = strcat('n=', num2str(n));
                                end

                            %end n for
                            end

                            %this needs to be outside n for, so can't be written in the else branch of if_variable_lambda before
                            if ~(if_variable_lambda)
                                % plot
                                title_n = strcat('train: t=', num2str(t_default), ' p=', num2str(p));
                                PlotCost(cost_n, 'savename', savename, 'legend', legend_n, 'title', title_n)
                                % save
                                % save(strcat('Dat/plot_', savename), 'cost', 'legend', 'title')
                                save(strcat('Dat/plot_', savename, '_n', num2str(n_count)), 'cost_n', 'legend_n','title_n')
                            end

                        %end p for
                        end

                    else
                        %%%%%%%%%plot Cost vs cost with different p
                        % load from example_train_p, legend
                        cost = cell(1,pnum);
                        legend = cell(1,pnum);

                        for p_count = 1:pnum
                            load(strcat('Dat/', savename, '_p', num2str(p_count), '.mat'), 'p', 'CList')
                            cost{p_count} = CList;
                            legend{p_count} = strcat('p=', num2str(p));
                        end
                        % plot
                        title = strcat('train: n=', num2str(0), ' t=', num2str(0.00));
                        PlotCost(cost, 'savename', savename, 'legend', legend, 'title', title)
                        % save
                        save(strcat('Dat/plot_', savename), 'cost', 'legend','title')
                    end

            otherwise
                error('Choose an example from 1 to 20.')
        end

    case 'test'
        switch example
            case {1, 2, 3, 4, 19, 20}
                m = 3;

                savename = strcat('dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test));

                % load from example_train_test
                load(strcat('Dat/', savename, '.mat'), 'n2', 'pnum', 't_number', 'n_number','if_variable_t','if_variable_n')

                                %plot cost vs iteration with different t

                if (if_variable_t)
                    % load from example_train_test_p
                    noise = zeros(1,t_number);
                    mfid_in = zeros(1,t_number);
                    mfid_out = zeros(1,t_number);
                    sdv_out = zeros(1,t_number);

                    for p_fixed_count = 1:pnum

                        for t_count = 1:t_number
                            if if_variable_lambda
                                noise = zeros(1,length(lambda_list));
                                mfid_in = zeros(1,length(lambda_list));
                                mfid_out = zeros(1,length(lambda_list));
                                sdv_out = zeros(1,length(lambda_list));

                                for lambda_count = 1:length(lambda_list)

                                    load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_t', num2str(t_count), '_lambda', num2str(lambda_count), '.mat'),...
                                        'p', 'meanfid_in', 'meanfid_out', 'varfid_out', 't', 'n_default','lambda')
                                    noise(lambda_count) = t;
                                    mfid_in(lambda_count) = meanfid_in;
                                    mfid_out(lambda_count) = meanfid_out;
                                    sdv_out(lambda_count) = sqrt(varfid_out);

                                end

                                % analytics
                                % [mfid_genin, varfid_genin, learnexpect] = FlipAnalyticsGHZ(3, noise);
                                [mfid_genin, varfid_genin] = FlipAnalyticsGHZ(m, noise);
                                sdv_genin = sqrt(varfid_genin / n2);
                                % divide genin into two data sets according to learnexpect:
                                % learnexpect = double(learnexpect);
                                % mask1 = learnexpect;
                                % mask1(mask1 == 0) = NaN;
                                % mask2 = 1 - learnexpect;
                                % mask2(mask2 == 0) = NaN;
                                % mfid_genin1 = mfid_genin .* mask1;
                                % mfid_genin2 = mfid_genin .* mask2;
                                % sdv_genin1 = sdv_genin .* mask1;
                                % sdv_genin2 = sdv_genin .* mask2;

                                % summarize
                                % fid = {mfid_genin1, mfid_genin2, mfid_in, mfid_out};
                                % sdv = {sdv_genin1, sdv_genin2, NaN * zeros(1,pnum), sdv_out};
                                % legend = ["gen. in 1", "gen. in 2", "noisy", "denoised"];

                                fid = {mfid_in, mfid_out};
                                sdv = {NaN * zeros(1,length(lambda_list)), sdv_out};
                                legend = ["noisy", "denoised"];

                                %%With gen. in , but for t not 0 wrong computed
                                %fid = {mfid_genin, mfid_in, mfid_out};
                                %sdv = {sdv_genin, NaN * zeros(1,t_number), sdv_out};
                                %legend = ["gen. in", "noisy", "denoised"];

                                title = strcat('n=', num2str(n_default), ' p=', num2str(p));
                                % plot
                                PlotDenoisingVsNoise(noise, fid,...
                                    'sdv_fidelity', sdv, 'savename', savename, 'legend', legend, 'title', title)

                                % save
                                save(strcat('Dat/plot_', savename, '_t', num2str(t_count), '_lambda'), 'noise', 'fid', 'sdv', 'legend', 'title')

                            else

                                load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_t', num2str(t_count), '.mat'),...
                                    'p', 'meanfid_in', 'meanfid_out', 'varfid_out', 't', 'n_default')
                                noise(t_count) = t;
                                mfid_in(t_count) = meanfid_in;
                                mfid_out(t_count) = meanfid_out;
                                sdv_out(t_count) = sqrt(varfid_out);

                            end

                        end

                        if ~(if_variable_lambda)

                            % analytics
                            % [mfid_genin, varfid_genin, learnexpect] = FlipAnalyticsGHZ(3, noise);
                            [mfid_genin, varfid_genin] = FlipAnalyticsGHZ(m, noise);
                            sdv_genin = sqrt(varfid_genin / n2);
                            % divide genin into two data sets according to learnexpect:
                            % learnexpect = double(learnexpect);
                            % mask1 = learnexpect;
                            % mask1(mask1 == 0) = NaN;
                            % mask2 = 1 - learnexpect;
                            % mask2(mask2 == 0) = NaN;
                            % mfid_genin1 = mfid_genin .* mask1;
                            % mfid_genin2 = mfid_genin .* mask2;
                            % sdv_genin1 = sdv_genin .* mask1;
                            % sdv_genin2 = sdv_genin .* mask2;

                            % summarize
                            % fid = {mfid_genin1, mfid_genin2, mfid_in, mfid_out};
                            % sdv = {sdv_genin1, sdv_genin2, NaN * zeros(1,pnum), sdv_out};
                            % legend = ["gen. in 1", "gen. in 2", "noisy", "denoised"];

                            fid = {mfid_in, mfid_out};
                            sdv = {NaN * zeros(1,t_number), sdv_out};
                            legend = ["noisy", "denoised"];

                            %%With gen. in , but for t not 0 wrong computed
                            %fid = {mfid_genin, mfid_in, mfid_out};
                            %sdv = {sdv_genin, NaN * zeros(1,t_number), sdv_out};
                            %legend = ["gen. in", "noisy", "denoised"];

                            title = strcat('n=', num2str(n_default), ' p=', num2str(p));
                            % plot
                            PlotDenoisingVsNoise(noise, fid,...
                                'sdv_fidelity', sdv, 'savename', savename, 'legend', legend, 'title', title)

                            % save
                            save(strcat('Dat/plot_', savename, '_t', num2str(t_count)), 'noise', 'fid', 'sdv', 'legend', 'title')

                        end

                    end

                elseif (if_variable_n)
                    noise = zeros(1,n_number);
                    mfid_in = zeros(1,n_number);
                    mfid_out = zeros(1,n_number);
                    sdv_out = zeros(1,n_number);

                    for p_fixed_count = 1:pnum

                        for n_count = 1:n_number
                            load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_n', num2str(n_count), '.mat'),...
                                'p', 'meanfid_in', 'meanfid_out', 'varfid_out', 'n', 't_default')
                            noise(n_count) = n;
                            mfid_in(n_count) = meanfid_in;
                            mfid_out(n_count) = meanfid_out;
                            sdv_out(n_count) = sqrt(varfid_out);
                        end


                        % analytics
                        % [mfid_genin, varfid_genin, learnexpect] = FlipAnalyticsGHZ(3, noise);
                        [mfid_genin, varfid_genin] = FlipAnalyticsGHZ(m, noise);
                        sdv_genin = sqrt(varfid_genin / n2);
                        % divide genin into two data sets according to learnexpect:
                        % learnexpect = double(learnexpect);
                        % mask1 = learnexpect;
                        % mask1(mask1 == 0) = NaN;
                        % mask2 = 1 - learnexpect;
                        % mask2(mask2 == 0) = NaN;
                        % mfid_genin1 = mfid_genin .* mask1;
                        % mfid_genin2 = mfid_genin .* mask2;
                        % sdv_genin1 = sdv_genin .* mask1;
                        % sdv_genin2 = sdv_genin .* mask2;

                        % summarize
                        % fid = {mfid_genin1, mfid_genin2, mfid_in, mfid_out};
                        % sdv = {sdv_genin1, sdv_genin2, NaN * zeros(1,pnum), sdv_out};
                        % legend = ["gen. in 1", "gen. in 2", "noisy", "denoised"];

                        fid = {mfid_in, mfid_out};
                        sdv = {NaN * zeros(1,n_number), sdv_out};
                        legend = ["noisy", "denoised"];

                        %%With gen. in , but for t not 0 wrong computed
                        %fid = {mfid_genin, mfid_in, mfid_out};
                        %sdv = {sdv_genin, NaN * zeros(1,n_number), sdv_out};
                        %legend = ["gen. in", "noisy", "denoised"];

                        % plot
                        title_n = strcat('t=', num2str(t_default), ' p=', num2str(p));
                        PlotDenoisingVsNoise(noise, fid,...
                            'sdv_fidelity', sdv, 'savename', savename, 'legend', legend, 'title', title_n)

                        % save
                        save(strcat('Dat/plot_', savename, '_n', num2str(n_count)), 'noise', 'fid', 'sdv', 'legend')

                    end

                else
                    % load from example_train_test_p
                    noise = zeros(1,pnum);
                    mfid_in = zeros(1,pnum);
                    mfid_out = zeros(1,pnum);
                    sdv_out = zeros(1,pnum);

                    for i = 1:pnum
                        load(strcat('Dat/', savename, '_p', num2str(i), '.mat'),...
                            'p', 'meanfid_in', 'meanfid_out', 'varfid_out')
                        noise(i) = p;
                        mfid_in(i) = meanfid_in;
                        mfid_out(i) = meanfid_out;
                        sdv_out(i) = sqrt(varfid_out);
                    end


                    % analytics
                    % [mfid_genin, varfid_genin, learnexpect] = FlipAnalyticsGHZ(3, noise);
                    [mfid_genin, varfid_genin] = FlipAnalyticsGHZ(m, noise);
                    sdv_genin = sqrt(varfid_genin / n2);
                    % divide genin into two data sets according to learnexpect:
                    % learnexpect = double(learnexpect);
                    % mask1 = learnexpect;
                    % mask1(mask1 == 0) = NaN;
                    % mask2 = 1 - learnexpect;
                    % mask2(mask2 == 0) = NaN;
                    % mfid_genin1 = mfid_genin .* mask1;
                    % mfid_genin2 = mfid_genin .* mask2;
                    % sdv_genin1 = sdv_genin .* mask1;
                    % sdv_genin2 = sdv_genin .* mask2;

                    % summarize
                    % fid = {mfid_genin1, mfid_genin2, mfid_in, mfid_out};
                    % sdv = {sdv_genin1, sdv_genin2, NaN * zeros(1,pnum), sdv_out};
                    % legend = ["gen. in 1", "gen. in 2", "noisy", "denoised"];

                    %%With gen. in , but for t not 0 wrong computed
                    %fid = {mfid_genin, mfid_in, mfid_out};
                    %sdv = {sdv_genin, NaN * zeros(1,pnum), sdv_out};
                    %legend = ["gen. in", "noisy", "denoised"];

                    fid = {mfid_in, mfid_out};
                    sdv = {NaN * zeros(1,pnum), sdv_out};
                    legend = ["noisy", "denoised"];

                    % plot
                    title = strcat('t=', num2str(0.00), ' p=', num2str(p));
                    PlotDenoisingVsNoise(noise, fid,...
                        'sdv_fidelity', sdv, 'savename', savename, 'legend', legend, 'title', title)

                    % save
                    save(strcat('Dat/plot_', savename), 'noise', 'fid', 'sdv', 'legend','title')

                end

            case {5, 6, 7, 8, 9, 10, 11}
                m = 4;

                savename = strcat('dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test));

                % load from example_train_test
                load(strcat('Dat/', savename, '.mat'), 'n2', 'pnum')

                % load from example_train_test_p
                noise = zeros(1,pnum);
                mfid_in = zeros(1,pnum);
                mfid_out = zeros(1,pnum);
                sdv_out = zeros(1,pnum);
                for i = 1:pnum
                    load(strcat('Dat/', savename, '_p', num2str(i), '.mat'),...
                        'p', 'meanfid_in', 'meanfid_out', 'varfid_out')
                    noise(i) = p;
                    mfid_in(i) = meanfid_in;
                    mfid_out(i) = meanfid_out;
                    sdv_out(i) = sqrt(varfid_out);
                end

                % analytics
                [mfid_genin, varfid_genin] = FlipAnalyticsGHZ(m, noise);
                sdv_genin = sqrt(varfid_genin / n2);

                % summarize
                fid = {mfid_genin, mfid_in, mfid_out};
                sdv = {sdv_genin, NaN * zeros(1,pnum), sdv_out};
                legend = ["gen. in", "noisy", "denoised"];

                % plot
                PlotDenoisingVsNoise(noise, fid,...
                    'sdv_fidelity', sdv, 'savename', savename, 'legend', legend)

                % save
                save(strcat('Dat/plot_', savename), 'noise', 'fid', 'sdv', 'legend')
            case {12, 13, 14, 15, 16, 17, 18}
                m = 5;

                savename = strcat('dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test));

                % load from example_train_test
                load(strcat('Dat/', savename, '.mat'), 'n2', 'pnum')

                % load from example_train_test_p
                noise = zeros(1,pnum);
                mfid_in = zeros(1,pnum);
                mfid_out = zeros(1,pnum);
                sdv_out = zeros(1,pnum);
                for i = 1:pnum
                    load(strcat('Dat/', savename, '_p', num2str(i), '.mat'),...
                        'p', 'meanfid_in', 'meanfid_out', 'varfid_out')
                    noise(i) = p;
                    mfid_in(i) = meanfid_in;
                    mfid_out(i) = meanfid_out;
                    sdv_out(i) = sqrt(varfid_out);
                end

                % analytics
                [mfid_genin, varfid_genin] = FlipAnalyticsGHZ(m, noise);
                sdv_genin = sqrt(varfid_genin / n2);

                % summarize
                fid = {mfid_genin, mfid_in, mfid_out};
                sdv = {sdv_genin, NaN * zeros(1,pnum), sdv_out};
                legend = ["gen. in", "noisy", "denoised"];

                % plot
                PlotDenoisingVsNoise(noise, fid,...
                    'sdv_fidelity', sdv, 'savename', savename, 'legend', legend)

                % save
                save(strcat('Dat/plot_', savename), 'noise', 'fid', 'sdv', 'legend')
            otherwise
                error('Choose an example from 1 to 20.')
        end
    otherwise
        error('Valid modes are "train" and "test".')
end

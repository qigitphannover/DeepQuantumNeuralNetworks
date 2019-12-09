
mode = 'test'; % 'train' or 'test'
example = 1;
id_train = 4;
id_test = 4;
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
            case {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
                savename = strcat('du_train_ex', num2str(example), '_train', num2str(id_train));

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

                                for lambda_count = 1:length(lambda_list)
                                    load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_t', num2str(t_count), '_lambda', num2str(lambda_count), '.mat'), 'p', 'CList','t', 'lambda', 'n_default')
                                    cost_lambda{lambda_count} = CList;
                                    legend_lambda{lambda_count} = strcat('lambda=', num2str(lambda));
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
                    cost_n = cell(1,n_number);
                    legend_n = cell(1, n_number);

                    for p_fixed_count = 1:pnum

                        for n_count = 1:t_number

                            if if_variable_lambda

                                for lambda_count = 1:length(lambda_list)
                                    load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_n', num2str(n_count), '_lambda', num2str(lambda_count), '.mat'), 'p', 'CList','n', 'lambda', 't_default')
                                    cost{lambda_count} = CList;
                                    legend{lambda_count} = strcat('lambda=', num2str(lambda));
                                end

                                % plot
                                title_lambda = strcat('train: n=', num2str(n), ' p=', num2str(p), ' t=', num2str(t_default));
                                PlotCost(cost_lambda, 'savename', savename, 'legend', legend_lambda, 'title', title_lambda)
                                % save
                                % save(strcat('Dat/plot_', savename), 'cost', 'legend', 'title')
                                save(strcat('Dat/plot_', savename, '_n', num2str(n_count), '_p', num2str(p_fixed_count)), 'cost_lambda', 'legend_lambda','title_lambda')

                            else

                                load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_n', num2str(n_count), '.mat'), 'p', 'CList','n', 't_default')
                                cost_n{n_count} = CList;
                                legend_n{n_count} = strcat('n=', num2str(n));

                            end

                        %end n for
                        end

                        %this needs to be outside t for, so can't be written in the else branch of if_variable_lambda before
                        if ~(if_variable_lambda)
                            % plot
                            title_t = strcat('train: n=', num2str(n_default), ' p=', num2str(p));
                            PlotCost(cost_n, 'savename', savename, 'legend', legend_n, 'title', title_n)
                            % save
                            % save(strcat('Dat/plot_', savename), 'cost', 'legend', 'title')
                            save(strcat('Dat/plot_', savename, '_n', num2str(n_count)), 'cost_n', 'legend_n','title_n')
                        end

                    %end p for
                    end

                else
                    if if_variable_lambda
                        for p_count = 1:pnum
                            %%%%%%%%%plot Cost vs cost with different p
                            % load from example_train_p, legend
                            cost = cell(1,length(lambda_list));
                            legend = cell(1,length(lambda_list));

                            for lambda_count = 1:length(lambda_list)
                                load(strcat('Dat/', savename, '_p', num2str(p_count), '_t', num2str(t_default), '_n', num2str(n_default), '_lambda', num2str(lambda_count), '.mat'), 'p', 'CList','t_default','n_default','lambda')
                                cost{lambda_count} = CList;
                                legend{lambda_count} = strcat('lambda=', num2str(lambda));
                            end

                            % plot
                            title = strcat('train: n=', num2str(n_default), ' p =', num2str(p), ' t=', num2str(t_default));
                            PlotCost(cost, 'savename', savename, 'legend', legend, 'title', title)

                            % save
                            save(strcat('Dat/plot_', savename, '_lambda', num2str(lambda_count)), 'cost', 'legend', 'title')

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
                        save(strcat('Dat/plot_', savename), 'cost', 'legend', 'title')
                    end

                end


            otherwise
                error('Choose an example from 1 to 20.')
        end

    case 'test'
        switch example
            case {1, 2, 3, 4, 19, 20}
                %%IMPORTANT: Use for test if to run old trained QNNs (if not, it will NOT run, or with old chached values!)
                %lambda = 0.0040
                %if_variable_lambda = false;
                %if_variable_t = true;
                %if_variable_n = false;
                %lambda_list = [0.0040, 0.0080, 0.0173];

                m = 3;

                savename = strcat('du_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test));

                % load from example_train_test
                load(strcat('Dat/', savename, '.mat'), 'n2', 'pnum', 't_number', 'n_number','if_variable_t','if_variable_n')

                if (if_variable_t)
                    % load from example_train_test_p
                    noise_t = zeros(1,t_number);
                    mfid_in_t = zeros(1,t_number);
                    mfid_out_t = zeros(1,t_number);
                    sdv_out_t = zeros(1,t_number);

                    for p_fixed_count = 1:pnum

                        if if_variable_lambda
                            % load from example_train_test_p
                            noise_lambda = zeros(1,t_number);
                            mfid_in_lambda = zeros(1,t_number);
                            mfid_out_lambda = zeros(1,t_number);
                            sdv_out_lambda = zeros(1,t_number);

                            for lambda_count = 1:length(lambda_list)
                                for t_count = 1:t_number

                                    load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_t', num2str(t_count), '_lambda', num2str(lambda_count), '.mat'),...
                                        'p', 'meanfid_in','varfid_in', 'meanfid_out', 'varfid_out', 't', 'n_default','lambda')
                                    noise_lambda(t_count) = t;
                                    mfid_in_lambda(t_count) = meanfid_in;
                                    sdv_in_lambda(t_count) = sqrt(varfid_in);
                                    mfid_out_lambda(t_count) = meanfid_out;
                                    sdv_out_lambda(t_count) = sqrt(varfid_out);

                                end

                                fid_lambda = { mfid_in_lambda, mfid_out_lambda};
                                sdv_lambda = { sdv_in_lambda, sdv_out_lambda};
                                legend_lambda = ["noisy", "denoised"];

                                % plot
                                title_lambda = strcat('n=', num2str(n_default), ' p=', num2str(p), ' lambda =', num2str(lambda));
                                PlotDenoisingVsNoise(noise_lambda, fid_lambda,...
                                    'sdv_fidelity', sdv_lambda, 'savename', savename, 'legend', legend_lambda, 'title', title_lambda)

                                % save
                                save(strcat('Dat/plot_', savename, '_t', num2str(t_count)), 'noise_lambda', 'fid_lambda', 'sdv_lambda', 'legend_lambda', 'title_lambda')

                            %lambda for end
                            end
                        else
                            for t_count = 1:t_number
                                load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_t', num2str(t_count), '.mat'),...
                                    'p', 'meanfid_in','varfid_in', 'meanfid_out', 'varfid_out', 't', 'n_default','lambda')
                                noise_t(t_count) = t;
                                mfid_in_t(t_count) = meanfid_in;
                                sdv_in_t(t_count) = sqrt(varfid_in);
                                mfid_out_t(t_count) = meanfid_out;
                                sdv_out_t(t_count) = sqrt(varfid_out);

                            end

                            % analytics
                            % [mfid_genin, varfid_genin, learnexpect] = FlipAnalyticsGHZ(3, noise);
                            % [mfid_genin, varfid_genin] = FlipAnalyticsGHZ(m, noise);
                            % sdv_genin = sqrt(varfid_genin / n2);
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
                            % fid = {mfid_genin, mfid_in, mfid_out};
                            % sdv = {sdv_genin, NaN * zeros(1,pnum), sdv_out};
                            % legend = ["gen. in", "noisy", "denoised"];


                            fid_t = { mfid_in_t, mfid_out_t};
                            sdv_t = { sdv_in_t, sdv_out_t};
                            legend_t = ["noisy", "denoised"];

                            % plot
                            title_t = strcat('n=', num2str(n_default), ' p=', num2str(p), ' lambda=', num2str(lambda));
                            PlotDenoisingVsNoise(noise_t, fid_t,...
                                'sdv_fidelity', sdv_t, 'savename', savename, 'legend', legend_t, 'title', title_t)

                            % save
                            save(strcat('Dat/plot_', savename, '_t', num2str(t_count)), 'noise_t', 'fid_t', 'sdv_t', 'legend_t', 'title_t', 'lambda')

                        end

                    %p for end
                    end

                elseif (if_variable_n)
                    % load from example_train_test_p
                    noise = zeros(1,n_number);
                    mfid_in = zeros(1,n_number);
                    mfid_out = zeros(1,n_number);
                    sdv_out = zeros(1,n_number);

                    for p_fixed_count = 1:pnum

                        if if_variable_lambda
                            % load from example_train_test_p
                            noise_lambda = zeros(1,n_number);
                            mfid_in_lambda = zeros(1,n_number);
                            mfid_out_lambda = zeros(1,n_number);
                            sdv_out_lambda = zeros(1,n_number);


                            for lambda_count = 1:length(lambda_list)
                                for n_count = 1:n_number
                                    load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_n', num2str(n_count), '_lambda', num2str(lambda_count), '.mat'),...
                                        'p', 'meanfid_in','varfid_in', 'meanfid_out', 'varfid_out', 'n', 't_default','lambda')
                                    noise_lamba(n_count) = n;
                                    mfid_in_lamba(n_count) = meanfid_in;
                                    sdv_in_lamba(n_count) = sqrt(varfid_in);
                                    mfid_out_lamba(n_count) = meanfid_out;
                                    sdv_out_lamba(n_count) = sqrt(varfid_out);
                                end

                                fid_lambda = { mfid_in, mfid_out};
                                sdv_lambda = { sdv_in, sdv_out};
                                legend_lambda = ["noisy", "denoised"];

                                % plot
                                title_lambda = strcat('t=', num2str(t_default), ' p=', num2str(p), ' lambda =', num2str(lambda));
                                PlotDenoisingVsNoise(noise_lambda, fid_lambda,...
                                    'sdv_fidelity', sdv_lambda, 'savename', savename, 'legend', legend_lambda, 'title', title_lambda)

                                % save
                                save(strcat('Dat/plot_', savename, '_n', num2str(n_count)), 'noise_lambda', 'fid_lambda', 'sdv_lambda', 'legend_lambda', 'title_lambda')


                            %lambda for end
                            end

                        else

                            for n_count = 1:n_number
                                load(strcat('Dat/', savename, '_p', num2str(p_fixed_count), '_n', num2str(n_count), '.mat'),...
                                    'p', 'meanfid_in','varfid_in', 'meanfid_out', 'varfid_out', 'n', 't_default')
                                noise_n(n_count) = n;
                                mfid_in_n(n_count) = meanfid_in;
                                sdv_in_n(n_count) = sqrt(varfid_in);
                                mfid_out_n(n_count) = meanfid_out;
                                sdv_out_n(n_count) = sqrt(varfid_out);
                            end

                            % analytics
                            % [mfid_genin, varfid_genin, learnexpect] = FlipAnalyticsGHZ(3, noise);
                            % [mfid_genin, varfid_genin] = FlipAnalyticsGHZ(m, noise);
                            % sdv_genin = sqrt(varfid_genin / n2);
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
                            % fid = {mfid_genin, mfid_in, mfid_out};
                            % sdv = {sdv_genin, NaN * zeros(1,pnum), sdv_out};
                            % legend = ["gen. in", "noisy", "denoised"];

                            fid_n = { mfid_in, mfid_out};
                            sdv_n = { sdv_in, sdv_out};
                            legend_n = ["noisy", "denoised"];

                            % plot
                            title_n = strcat('t=', num2str(t_default), ' p=', num2str(p));
                            PlotDenoisingVsNoise(noise_n, fid_n,...
                                'sdv_fidelity', sdv_n, 'savename', savename, 'legend', legend_n, 'title', title_n)

                            % save
                            save(strcat('Dat/plot_', savename, '_n', num2str(n_count)), 'noise_n', 'fid_n', 'sdv_n', 'legend_n', 'title_n')

                        end

                    %p for end
                    end

                else

                    if if_variable_lambda
                        for lambda_count = 1:length(lambda_list)
                            % load from example_train_test_p
                            noise = zeros(1,pnum);
                            mfid_in = zeros(1,pnum);
                            mfid_out = zeros(1,pnum);
                            sdv_out = zeros(1,pnum);
                            for p_count = 1:pnum
                                load(strcat('Dat/', savename, '_p', num2str(p_count), '_t', num2str(t_default), '_n', num2str(n_default), '_lambda', num2str(lambda_count), '.mat'),...
                                    'p', 'meanfid_in','varfid_in', 'meanfid_out', 'varfid_out','t_default','n_default','lambda')
                                noise(p_count) = p;
                                mfid_in(p_count) = meanfid_in;
                                sdv_in(p_count) = sqrt(varfid_in);
                                mfid_out(p_count) = meanfid_out;
                                sdv_out(p_count) = sqrt(varfid_out);
                            end

                            fid = { mfid_in, mfid_out};
                            sdv = { sdv_in, sdv_out};
                            legend = ["noisy", "denoised"];

                            % plot
                            title = strcat('t=', num2str(t_default), ' n=', num2str(n_default), ' lambda =', num2str(lambda));
                            PlotDenoisingVsInputDataError(noise, fid,...
                                'sdv_fidelity', sdv, 'savename', savename, 'legend', legend, 'title', title)

                            % save
                            save(strcat('Dat/plot_', savename, '_lambda', num2str(lambda_count)), 'noise', 'fid', 'sdv', 'legend', 'title')

                        end
                    else
                        % load from example_train_test_p
                        noise = zeros(1,pnum);
                        mfid_in = zeros(1,pnum);
                        mfid_out = zeros(1,pnum);
                        sdv_out = zeros(1,pnum);
                        for i = 1:pnum
                            load(strcat('Dat/', savename, '_p', num2str(i), '.mat'),...
                                'p', 'meanfid_in','varfid_in', 'meanfid_out', 'varfid_out')
                            noise(i) = p;
                            mfid_in(i) = meanfid_in;
                            sdv_in(i) = sqrt(varfid_in);
                            mfid_out(i) = meanfid_out;
                            sdv_out(i) = sqrt(varfid_out);
                        end

                        % analytics
                        % [mfid_genin, varfid_genin, learnexpect] = FlipAnalyticsGHZ(3, noise);
                        % [mfid_genin, varfid_genin] = FlipAnalyticsGHZ(m, noise);
                        % sdv_genin = sqrt(varfid_genin / n2);
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
                        % fid = {mfid_genin, mfid_in, mfid_out};
                        % sdv = {sdv_genin, NaN * zeros(1,pnum), sdv_out};
                        % legend = ["gen. in", "noisy", "denoised"];

                        fid = { mfid_in, mfid_out};
                        sdv = { sdv_in, sdv_out};
                        legend = ["noisy", "denoised"];

                        % plot
                        title = strcat('t=', num2str(0.00));
                        PlotDenoisingVsInputDataError(noise, fid,...
                            'sdv_fidelity', sdv, 'savename', savename, 'legend', legend, 'title', title)

                        % save
                        save(strcat('Dat/plot_', savename), 'noise', 'fid', 'sdv', 'legend', 'title')

                    end

                end

            case {5, 6, 7, 8, 9, 10, 11}
                m = 4;

                savename = strcat('du_test_ex', num2str(example), '_train',...
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

                savename = strcat('du_test_ex', num2str(example), '_train',...
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

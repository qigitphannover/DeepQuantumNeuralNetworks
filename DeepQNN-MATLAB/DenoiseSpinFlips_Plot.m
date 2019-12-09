
mode = 'train'; % 'train' or 'test'
example = 10;
id_train = 14;
id_test = 1;
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
% 21: [3,1,3,1,3] denoising GHZ with random phase, training [3,1,3]
% 22: [3,2,1,3] denoising GHZ with random phase
% 23: [3,2,1,2,3] denoising GHZ with random phase
% 24: [3,2,3] denoising GHZ with random phase
% 25: [3,2,2,1,3,1,3] denoising GHZ with random phase
% 26: [3,2,2,2,3] denoising GHZ with random phase
% 27: [3,4,2,1,3] denoising GHZ with random phase

switch mode
    case 'train'
        switch example
            case {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28}
                savename = strcat('dsf_train_ex', num2str(example), '_train', num2str(id_train));
                
                % load from example_train (title)
                load(strcat('Dat/', savename, '.mat'), 'pnum')   
                % title = strcat('n=', num2str(n1), ', l=', num2str(lambda));
                
                % load from example_train_p, legend
                cost = cell(1,pnum);
                legend = cell(1,pnum);
                for i = 1:pnum
                    load(strcat('Dat/', savename, '_p', num2str(i), '.mat'), 'p', 'CList')
                    cost{i} = CList;
                    legend{i} = strcat('p=', num2str(p));
                end
                
                % plot
                % PlotCost(cost, 'savename', savename, 'legend', legend, 'title', title)
                PlotCost(cost, 'savename', savename, 'legend', legend)
                
                % save
                % save(strcat('Dat/plot_', savename), 'cost', 'legend', 'title')
                save(strcat('Dat/plot_', savename), 'cost', 'legend')     
            case 29
                savename = strcat('dsf_train_ex', num2str(example), '_train', num2str(id_train));
                
                % load from example_train 
                load(strcat('Dat/', savename, '.mat'), 'p', 'CList')   
                cost = CList;
                legend = strcat('p=', num2str(p));
                                
                % plot
                % PlotCost(cost, 'savename', savename, 'legend', legend, 'title', title)
                PlotCost(cost, 'savename', savename, 'legend', legend)
                
                % save
                % save(strcat('Dat/plot_', savename), 'cost', 'legend', 'title')
                save(strcat('Dat/plot_', savename), 'cost', 'legend')                   
            otherwise
                error('Choose an example from 1 to 27.')               
        end
        
    case 'test'
        switch example
            case {1, 2, 3, 4, 19, 21, 22, 23, 24, 25, 26, 27, 28}
                m = 3;
                
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
                fid = {mfid_genin, mfid_in, mfid_out};
                sdv = {sdv_genin, NaN * zeros(1,pnum), sdv_out};
                % legend = ["gen. in", "noisy", "denoised"];
                legend = ["expected noise", "noisy sample", "[3,1,3] denoised"];
                
                % plot
                PlotDenoisingVsNoise(noise, fid,...
                    'sdv_fidelity', sdv, 'savename', savename, 'legend', legend,...
                    'xlabel', 'spin-flip probability p', 'ylabel', 'fidelity with GHZ-0/GHZ-$\pi$',...
                    'plotstyle', {{"+",'MarkerSize', 5, 'LineWidth', 1.5,'CapSize',0},...
                    {".", 'MarkerSize', 25, 'LineWidth', 1.5},...
                    {"o", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 20}})     
                
                % save
                save(strcat('Dat/plot_', savename), 'noise', 'fid', 'sdv', 'legend')
            case 20
                m = 3;
                
                savename = strcat('dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test));
                
                % load from example_train_test
                load(strcat('Dat/', savename, '.mat'), 'n2', 'pnum')
                
                % load from example_train_test_p
                noise = zeros(1,pnum);
                mfid_in = zeros(1,pnum);
                mfid_inf = zeros(1,pnum);
                mfid_out = zeros(1,pnum);
                mfid_outf = zeros(1,pnum);
                sdv_out = zeros(1,pnum);
                sdv_outf = zeros(1,pnum);
                for i = 1:pnum
                    load(strcat('Dat/', savename, '_p', num2str(i), '.mat'),...
                        'p',  'meanfid_in', 'meanfid_inf',...
                        'meanfid_out', 'meanfid_outf', 'varfid_out', 'varfid_outf') 
                    noise(i) = p;
                    mfid_in(i) = meanfid_in;
                    mfid_inf(i) = meanfid_inf;
                    mfid_out(i) = meanfid_out;
                    sdv_out(i) = sqrt(varfid_out);
                    mfid_outf(i) = meanfid_outf;
                    sdv_outf(i) = sqrt(varfid_outf);                    
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
                fid = {mfid_in, mfid_inf, mfid_out, mfid_outf};
                sdv = {NaN * zeros(1,pnum), NaN * zeros(1,pnum), sdv_out, sdv_outf};
                % legend = ["gen. in", "noisy", "denoised"];
                legend = ["noisy", "noisy, max. 1 flip", "[3,1,3] denoised", "[3,1,3] denoised, \newlinemax. 1 flip"];
                
                % plot
                PlotDenoisingVsNoise(noise, fid,...
                    'sdv_fidelity', sdv, 'savename', savename, 'legend', legend,...
                    'xlabel', 'spin-flip probability p', 'ylabel', 'fidelity with GHZ-$\phi$',...
                    'plotstyle', {{".",'MarkerSize', 25, 'LineWidth', 1.5, 'Color', [0.85,0.325,0.098]},...
                    {"s", 'MarkerSize', 7, 'LineWidth', 1.5, 'MarkerFaceColor', [0, 0.447, 0.741]},...
                    {"o", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 20, 'Color', [0.929,0.694,0.125]},...
                    {"x", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 15, 'Color', [0.494,0.184,0.556]}})%,...
                    %'lposition', [0.36,0.21])     
                
                % save
                save(strcat('Dat/plot_', savename), 'noise', 'fid', 'sdv', 'legend')                
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
                    'sdv_fidelity', sdv, 'savename', savename, 'legend', legend,...
                    'plotstyle', {{"+",'MarkerSize', 5, 'LineWidth', 1.5,'CapSize',0},...
                    {".", 'MarkerSize', 25, 'LineWidth', 1.5},...
                    {".", 'MarkerSize', 25, 'LineWidth', 1.5}})     
                
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
            case 29
                m = 4;
                
                savename = strcat('dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test));
                ylabel = 'fidelity with GHZ-0';
                legend = {'noisy', '[4,2,1,2,4] denoised'};
                lposition = [0.71, 0.7];
                
                % load from example_train_test
                load(strcat('Dat/', savename, '.mat'), 'p', 'fid_in', 'fid_out')

                % plot
                PlotDenoising(fid_in, fid_out, 'savename', savename,...
                    'ylabel', ylabel, 'legend', legend, 'lposition', lposition)
                
                % save
                save(strcat('Dat/plot_', savename), 'fid_in', 'fid_out', 'ylabel', 'legend', 'lposition')                
            otherwise
                error('Choose an example from 1 to 27.')
        end        
    otherwise
        error('Valid modes are "train" and "test".')        
end
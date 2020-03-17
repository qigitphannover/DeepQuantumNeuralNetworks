
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
            case {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,...
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40}
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
                error('Choose an example from 1 to 40.')               
        end
        
    case 'test'
        switch example
            case {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28}
                savename = strcat('dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test));
                
                % load from example_train_test
                load(strcat('Dat/', savename, '.mat'), 'n2', 'pnum', 'M')
                m = M(1);
                
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
                legend = ["expected noise", "noisy sample", "denoised"];
                % legend = ["expected noise", "noisy sample", "[3,1,3] denoised"];
                
                % plot
                PlotDenoisingVsNoise(noise, fid,...
                    'sdv_fidelity', sdv, 'savename', savename, 'legend', legend,...
                    'xlabel', 'spin-flip probability p', 'ylabel', 'fidelity with GHZ state',... % 'GHZ-0' or 'GHZ-0/GHZ-$\pi$'
                    'plotstyle', {{"+", 'MarkerSize', 5, 'LineWidth', 1.5, 'CapSize', 0},...
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

                fid = {mfid_in, mfid_inf, mfid_out, mfid_outf};
                sdv = {NaN * zeros(1,pnum), NaN * zeros(1,pnum), sdv_out, sdv_outf};
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
             case {30, 31, 36}
                savename = strcat('dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test));
                
                % load from example_train_test
                load(strcat('Dat/', savename, '.mat'), 'n2', 'pnum', 'M')
                m = M(1);
                k = 1;
                
                % load from example_train_test_p
                noise = zeros(1,pnum);
                mfid_in = zeros(1,pnum);
                % sdv_in = zeros(1,pnum);
                mfid_out = zeros(1,pnum);
                sdv_out = zeros(1,pnum);
                for i = 1:pnum
                    load(strcat('Dat/', savename, '_p', num2str(i), '.mat'),...
                        'p', 'meanfid_in', 'varfid_in', 'meanfid_out', 'varfid_out') 
                    noise(i) = p;
                    mfid_in(i) = meanfid_in;
                    % sdv_in(i) = sqrt(varfid_in);
                    mfid_out(i) = meanfid_out;
                    sdv_out(i) = sqrt(varfid_out);
                end
                
                % analytics
                [mfid_genin, varfid_genin] = FlipAnalyticsDicke(m, k, noise);
                sdv_genin = sqrt(varfid_genin / n2);
                
                % summarize
                fid = {mfid_genin, mfid_in, mfid_out};
                sdv = {sdv_genin, NaN * zeros(1,pnum), sdv_out};
                legend = ["expected noise", "noisy sample", "denoised"];                
                % legend = ["expected noise", "noisy sample", "[3,1,3] denoised"];
                
                % plot
                PlotDenoisingVsNoise(noise, fid,...
                    'sdv_fidelity', sdv, 'savename', savename, 'legend', legend,...
                    'xlabel', 'spin-flip probability p', 'ylabel', 'fidelity with $D_1^3$',... % 'Dicke state'
                    'plotstyle', {{"+", 'MarkerSize', 5, 'LineWidth', 1.5, 'CapSize', 0},...
                    {".", 'MarkerSize', 25, 'LineWidth', 1.5},...
                    {"o", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 20}})
                    % {{".", 'MarkerSize', 25, 'LineWidth', 1.5, 'Color', [0.85,0.325,0.098]},...
                    % {"o", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 20, 'Color', [0.929,0.694,0.125]}})      
                    %'lposition', [0.25,0.235])

                % save
                save(strcat('Dat/plot_', savename), 'noise', 'fid', 'sdv', 'legend')
             case {32, 33, 34, 35}
                savename = strcat('dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test));
                
                % load from example_train_test
                load(strcat('Dat/', savename, '.mat'), 'n2', 'pnum', 'M')
                m = M(1);
                k = 2;
                
                % load from example_train_test_p
                noise = zeros(1,pnum);
                mfid_in = zeros(1,pnum);
                % sdv_in = zeros(1,pnum);
                mfid_out = zeros(1,pnum);
                sdv_out = zeros(1,pnum);
                for i = 1:pnum
                    load(strcat('Dat/', savename, '_p', num2str(i), '.mat'),...
                        'p', 'meanfid_in', 'varfid_in', 'meanfid_out', 'varfid_out') 
                    noise(i) = p;
                    mfid_in(i) = meanfid_in;
                    % sdv_in(i) = sqrt(varfid_in);
                    mfid_out(i) = meanfid_out;
                    sdv_out(i) = sqrt(varfid_out);
                end
                
                % analytics
                [mfid_genin, varfid_genin] = FlipAnalyticsDicke(m, k, noise);
                sdv_genin = sqrt(varfid_genin / n2);
                
                % summarize
                fid = {mfid_genin, mfid_in, mfid_out};
                sdv = {sdv_genin, NaN * zeros(1,pnum), sdv_out};
                legend = ["expected noise", "noisy sample", "[" + regexprep(num2str(M), '\s+', ',')+ "] denoised"];
                
                % plot
                PlotDenoisingVsNoise(noise, fid,...
                    'sdv_fidelity', sdv, 'savename', savename, 'legend', legend,...
                    'xlabel', 'spin-flip probability p', 'ylabel', 'fidelity with $|2,2\rangle$',... % 'Dicke state'
                    'plotstyle', {{"+", 'MarkerSize', 5, 'LineWidth', 1.5, 'CapSize', 0},...
                    {".", 'MarkerSize', 25, 'LineWidth', 1.5},...
                    {"o", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 20}})
                    % {{".", 'MarkerSize', 25, 'LineWidth', 1.5, 'Color', [0.85,0.325,0.098]},...
                    % {"o", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 20, 'Color', [0.929,0.694,0.125]}})      
                    %'lposition', [0.25,0.235])
                
                % save
                save(strcat('Dat/plot_', savename), 'noise', 'fid', 'sdv', 'legend')
            case {37, 38, 39, 40}
                savename = strcat('dsf_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test));
                
                % load from example_train_test
                load(strcat('Dat/', savename, '.mat'), 'n2', 'pnum', 'ideal')
                
                % load from example_train_test_p
                noise = zeros(1,pnum);
                mfid_in = zeros(1,pnum);
                sdv_in = zeros(1,pnum);
                mfid_out = zeros(1,pnum);
                sdv_out = zeros(1,pnum);
                for i = 1:pnum
                    load(strcat('Dat/', savename, '_p', num2str(i), '.mat'),...
                        'p', 'meanfid_in', 'varfid_in', 'meanfid_out', 'varfid_out') 
                    noise(i) = p;
                    mfid_in(i) = meanfid_in;
                    sdv_in(i) = sqrt(varfid_in);
                    mfid_out(i) = meanfid_out;
                    sdv_out(i) = sqrt(varfid_out);
                end
                
                % analytics
                [mfid_genin, varfid_genin] = FlipAnalytics(ideal, noise);
                sdv_genin = sqrt(varfid_genin / n2);
                
                % summarize
                fid = {mfid_genin, mfid_in, mfid_out};
                % fid = {mfid_in, mfid_out};
                sdv = {sdv_genin, NaN * zeros(1,pnum), sdv_out};
                % sdv = {sdv_in, sdv_out};
                legend = ["expected noise", "noisy sample", "[" + regexprep(num2str(M), '\s+', ',')+ "] denoised"];
                % legend = ["noisy", "[" + regexprep(num2str(M), '\s+', ',')+ "] denoised"];
                
                % plot
                PlotDenoisingVsNoise(noise, fid,...
                    'sdv_fidelity', sdv, 'savename', savename, 'legend', legend,...
                    'xlabel', 'spin-flip probability p', 'ylabel', 'fidelity with G$_{sq}$',... 
                    'plotstyle',... 
                    {{"+", 'MarkerSize', 5, 'LineWidth', 1.5, 'CapSize', 0},...
                    {".", 'MarkerSize', 25, 'LineWidth', 1.5},...
                    {"o", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 20}})
                    % {{".", 'MarkerSize', 25, 'LineWidth', 1.5, 'Color', [0.85,0.325,0.098]},...
                    % {"o", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 20, 'Color', [0.929,0.694,0.125]}})      
                    % 'lposition', [0.25,0.235])
                
                % save
                save(strcat('Dat/plot_', savename), 'noise', 'fid', 'sdv', 'legend')
            otherwise
                error('Choose an example from 1 to 40.')
        end        
    otherwise
        error('Valid modes are "train" and "test".')        
end
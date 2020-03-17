
mode = 'test'; % 'train' or 'test'
example = 39;
id_train = 4;
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

% W-states Dicke(3,1)
% 30: [3,1,3]
% 31: [3,1,3,1,3] training [3,1,3]

% Dicke states Dicke(4,2)
% 32: [4,1,4]
% 33: [4,1,4,1,4] training [4,1,4]
% 34: [4,2,1,2,4]
% 35: [4,2,1,2,4] sparse

switch mode
    case 'train'
        switch example
            case {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,... 
                    30, 31, 32, 33, 34, 35}
                savename = strcat('dp_train_ex', num2str(example), '_train', num2str(id_train));
                
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
            otherwise
                error('Choose an example from 1 to 20 or 30 to 35.')               
        end
        
    case 'test'
        switch example
            case {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,...
                    30, 31, 32, 33, 34, 35}
                
                savename = strcat('dp_test_ex', num2str(example), '_train',...
                    num2str(id_train), '_test', num2str(id_test));
                
                % load from example_train_test
                load(strcat('Dat/', savename, '.mat'), 'n2', 'pnum')
                
                % load from example_train_test_p
                noise = zeros(1,pnum);
                mfid_in = zeros(1,pnum);
                sdv_in = zeros(1,pnum);
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
                              
                fid = {mfid_in, mfid_out};
                sdv = {sdv_in, sdv_out};
                legend = ["noisy", "denoised"];
                
                % plot
                PlotDenoisingVsNoise(noise, fid,...
                    'sdv_fidelity', sdv, 'savename', savename, 'legend', legend, 'lposition', [0.25,0.235],...
                    'xlabel', 'phase spread $\sigma/\pi$', 'ylabel', 'fidelity with GHZ-0',...
                    'plotstyle', {{".", 'MarkerSize', 25, 'LineWidth', 1.5, 'Color', [0.85,0.325,0.098]},...
                    {"o", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 20, 'Color', [0.929,0.694,0.125]}})                
                
                % save
                save(strcat('Dat/plot_', savename), 'noise', 'fid', 'sdv', 'legend')
                
            otherwise
                error('Choose an example from 1 to 20 or 30 to 35.')
        end        
    otherwise
        error('Valid modes are "train" and "test".')        
end
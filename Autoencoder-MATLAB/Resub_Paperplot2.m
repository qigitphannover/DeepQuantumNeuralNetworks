
savename2 = 'resub_pp2';
load('Dat/plot_dp_test_ex8_train4_test3', 'noise', 'fid', 'sdv')
fid2 = fid;
sdv2 = sdv;
load('Dat/plot_dp_test_ex10_train5_test2', 'fid', 'sdv')
noise2 = noise;
fid2 = cat(2, fid2, fid{end});
sdv2 = cat(2, sdv2, sdv{end});
% legend2 = ["noisy", "[4,2,1,2,4] \newlinedenoised", "[4,1,4,1,4] \newlinedenoised"];
legend2 = ["noisy", "[4,2,1,2,4] denoised", "[4,1,4,1,4] denoised"];

PlotDenoisingVsNoise(noise2, fid2,...
    'sdv_fidelity', sdv2, 'savename', savename2, 'legend', legend2,...
    'xlabel', 'phase spread $\sigma / \pi$', 'ylabel', 'fidelity with GHZ-0',...
    'plotstyle', {{".", 'MarkerSize', 25, 'LineWidth', 1.5, 'Color', [0.85,0.325,0.098]},...
    {"o", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 20, 'Color', [0.929,0.694,0.125]},...
    {"x", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 15, 'Color', [0.494,0.184,0.556]}})
    % 'lposition', [0.247,0.23],
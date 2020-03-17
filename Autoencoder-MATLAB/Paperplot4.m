
savename1 = 'pp4_b';
load('Dat/plot_dsf_test_ex10_train13_test1', 'noise', 'fid', 'sdv')
fid1 = {fid{2}, fid{1}, fid{3}};
sdv1 = {sdv{2}, sdv{1}, sdv{3}};
noise1 = noise;
legend1 = ["noisy sample", "expected noise", "[4,1,4,1,4] denoised"];

PlotDenoisingVsNoise(noise1, fid1,...
    'sdv_fidelity', sdv1, 'savename', savename1, 'legend', legend1, 'lposition', [0.745,0.852],...
    'xlabel', 'spin-flip probability p', 'ylabel', 'fidelity with GHZ-0',...
    'plotstyle', {{".", 'MarkerSize', 25, 'LineWidth', 1.5, 'Color', [0.85,0.325,0.098]},...
    {"+",'MarkerSize', 5, 'LineWidth', 1.5, 'CapSize', 0, 'Color', [0, 0.447, 0.741]},...
    {"x", 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 15, 'Color', [0.494,0.184,0.556]}})

ax = gca;
ax.XLim = [0.325, 0.49];
ax.XTick = [0.35, 0.4, 0.45];
savefig(strcat('Fig/dnvn_', savename1, '.fig'))
saveas(gcf, strcat('Fig/dnvn_', savename1), 'png')
saveas(gcf, strcat('Fig/dnvn_', savename1), 'epsc')
function PlotDenoising(fidelity_in, fidelity_out, varargin)
% PlotDenoising plots the ideal state fidelity of noisy and denoised states
%
% in: 
% fidelity_in: array of noisy state fidelities
% fidelity_out: array of denoised state fidelities
% optional in as name/value pairs (function(..., 'name', value, ...)):
% savename: plots are saved at 'Fig/dn_savename'; default is no saving
% title: a plot title; default is no title

if rem(length(varargin), 2) == 1 % test that optional input is paired
    error('Provide the optional input arguments as name/value pairs.')
end
varargin = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);

figure % plot
hold on 

plot(fidelity_in, 'rd', 'LineWidth', 2.5)
plot(fidelity_out, 'b+', 'LineWidth', 2.5)

% set plot properties
legend('noisy', 'denoised', 'Orientation', 'horizontal', 'Location', 'southeast')
ax = gca;
ax.YLim = [0, 1.1]; 
ax.LineWidth = 1.5;
ax.FontSize = 16; % ticks; if not explicitly set: labels, legend, title
xlabel('sample')
ylabel('fidelity')
if isfield(varargin, 'title')
    title(varargin.('title'), 'FontWeight', 'normal')
end
hold off

if isfield(varargin, 'savename') % save
    savename = varargin.('savename');
    savefig(strcat('Fig/dn_', savename, '.fig'))
    saveas(gcf, strcat('Fig/dn_', savename), 'png')
    saveas(gcf, strcat('Fig/dn_', savename), 'epsc')
end
end
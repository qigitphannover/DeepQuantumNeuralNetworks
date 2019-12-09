function PlotDenoising(fidelity_in, fidelity_out, varargin)
% PlotDenoising plots the ideal state fidelity of noisy and denoised states
%
% in: 
% fidelity_in: array of noisy state fidelities
% fidelity_out: array of denoised state fidelities
% optional in as name/value pairs (function(..., 'name', value, ...)):
% savename: plots are saved at 'Fig/dn_savename'; default is no saving
% title: a plot title; default is no title
% ylabel: label string; default is 'fidelity'
% legend: a cell of legend labels;
%         default: {'noisy', 'denoised'}
% lposition: [left, bottom] array for legend positioning;
%            default is 'best'

if rem(length(varargin), 2) == 1 % test that optional input is paired
    error('Provide the optional input arguments as name/value pairs.')
end
varargin = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);

figure % plot
hold on 

plot(fidelity_in, "o", 'MarkerSize', 4, 'LineWidth', 1.5, 'Color', [0.85,0.325,0.098], 'MarkerFaceColor', [0.85,0.325,0.098])
plot(fidelity_out, "o", 'MarkerSize', 4, 'LineWidth', 1.5, 'Color', [0.929,0.694,0.125])

% set plot properties
if isfield(varargin, 'legend')
    label = varargin.('legend');
else
    label = {'noisy', 'denoised'};
end
if isfield(varargin, 'lposition')
    pos = {'Position', cat(2, varargin.('lposition'), [0,0])};
else 
    pos = {};
end
legend(label{:}, 'Orientation', 'horizontal', 'NumColumns', 1,...
    'Location', 'best', pos{:}) 
ax = gca;
ax.YLim = [0, 1.1]; 
ax.LineWidth = 1.5;
ax.FontSize = 12; % ticks; if not explicitly set: labels, legend, title
xlabel('test no')
if isfield(varargin, 'ylabel')
    ylabel(varargin.('ylabel'))
else
    ylabel('fidelity')
end
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
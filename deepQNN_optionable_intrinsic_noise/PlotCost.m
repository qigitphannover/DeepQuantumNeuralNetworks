function PlotCost(cost, varargin)
% PlotCost plots cost arrays as a function of training round
%
% in:
% cost: cost array or cell of cost arrays
% optional in as name/value pairs (function(..., 'name', value, ...)):
% savename: plots are saved at 'Fig/cost_savename'; default is no saving
% legend: a cell of legend labels;
%         default: 1, 2,... if more than one data set is drawn
% title: a plot title; default is no title

if rem(length(varargin), 2) == 1 % test that optional input is paired
    error('Provide the optional input arguments as name/value pairs.')
end
varargin = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);

if ~iscell(cost) % for a single data set: ensure cost is a cell
   cost = {cost}; 
end
num = length(cost);

figure % plot
hold on

for i = 1:num
    plot(cost{i}, 'LineWidth', 2.5)
end

% set plot properties
if isfield(varargin, 'legend') || num > 1
    if isfield(varargin, 'legend')
        leg = varargin.('legend');
    else
        leg = string(1:num);        
    end
    legend(leg, 'Location', 'eastoutside') 
end
ax = gca;
ax.YLim = [0, 1.1]; 
ax.LineWidth = 1.5;
ax.FontSize = 16; % ticks; if not explicitly set: labels, legend
xlabel('iteration')
ylabel('fidelity')
if isfield(varargin, 'title')
    title(varargin.('title'), 'FontWeight', 'normal')
end
hold off

if isfield(varargin, 'savename') % save
    savename = varargin.('savename');
    savefig(strcat('Fig/cost_', savename, '.fig'))
    saveas(gcf, strcat('Fig/cost_', savename), 'png')
    saveas(gcf, strcat('Fig/cost_', savename), 'epsc')
end
end
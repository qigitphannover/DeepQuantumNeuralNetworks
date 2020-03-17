function PlotDenoisingVsNoise(noise, mean_fidelity, varargin)
% PlotDenoisingVsNoise plots mean fidelities as a function of noise strength
%
% in:
% noise: an array of noise strengths or a cell of noise strength arrays
% mean_fidelity: corr. array of mean fidelities or a cell of arrays;
%                if all cell elements correspond to one noise array,
%                noise can be provided as one array
% optional in as name/value pairs (function(..., 'name', value, ...)):
% sdv_fidelity: an array or cell of fidelity standard deviations
% savename: plots are saved at 'Fig/dnvn_savename'; default is no saving
% legend: a cell of legend labels;
%         default: 1, 2,... if more than one data set is drawn
% plotstyle: a cell of plot styles as input to the errorbar function;
%            combine more than one prescription per plot style into a cell;
%            if only one data set is plotted, plotstyle can also be 
%            a string or a cell directly containing several prescriptions;
%            default: {".", 'MarkerSize', 25, 'LineWidth', 1.5}
% xlabel: label string; default is 'noise strength'
% lposition: [left, bottom] array for legend positioning;
%            default is 'southwest'

if rem(length(varargin), 2) == 1 % test that optional input is paired
    error('Provide the optional input arguments as name/value pairs.')
end
varargin = cell2struct(varargin(2:2:end), varargin(1:2:end), 2);

if ~iscell(mean_fidelity) % for a single data set: ensure mean_fidelity is a cell
   mean_fidelity = {mean_fidelity}; 
end
num = length(mean_fidelity);

if isfield(varargin, 'sdv_fidelity') % define error bars
    err = varargin.('sdv_fidelity');
    if ~iscell(err) % for a single data set: ensure err is a cell
        err = {err};
    end
else
    err = cellfun(@(x) NaN*x, mean_fidelity, 'UniformOutput', 0);
end
if isfield(varargin, 'plotstyle') % define plot styles
    plotstyle = varargin.('plotstyle');
    if num == 1 % flexible input parsing for single data set
        if iscell(plotstyle)
            if ~iscell(plotstyle{1})
                plotstyle = {plotstyle};
            end
        else
            plotstyle = {{plotstyle}};
        end
    else
        if iscell(plotstyle)
            for i = 1:num % flexible input parsing for multiple data sets
                istyle = plotstyle{i};
                if ~iscell(istyle)
                    plotstyle{i} = {istyle};
                end
            end
        else
            error('The plotstyle has to be a cell of styles for each data set.')
        end
    end        
else
    plotstyle = cell(1,num); % default plot style
    [plotstyle{:}] = deal({".", 'MarkerSize', 25, 'LineWidth', 1.5});
end

figure % plot
hold on

if iscell(noise)
    for i = 1:num
        errorbar(noise{i}, mean_fidelity{i}, err{i}, plotstyle{i}{:})
    end
else
    for i = 1:num
        errorbar(noise, mean_fidelity{i}, err{i}, plotstyle{i}{:})
    end
end

% set plot properties
if isfield(varargin, 'legend') || num > 1
    if isfield(varargin, 'legend')
        leg = varargin.('legend');
    else
        leg = string(1:num);        
    end
    if isfield(varargin, 'lposition')
        lpos = varargin.('lposition');
        legend(leg, 'Position', [lpos(1),lpos(2),0,0])
    else
        legend(leg, 'Location', 'southwest')
    end
end
ax = gca;
ax.YLim = [0, 1.15]; 
ax.LineWidth = 1.5;
ax.FontSize = 12; % ticks; if not explicitly set: labels, legend
if isfield(varargin, 'xlabel')
    xlabel(varargin.('xlabel'))
else
    xlabel('noise strength')
end
if isfield(varargin, 'ylabel')
    ylabel(varargin.('ylabel'))
else
    ylabel('mean fidelity')
end
hold off

if isfield(varargin, 'savename') % save
    savename = varargin.('savename');
    savefig(strcat('Fig/dnvn_', savename, '.fig'))
    saveas(gcf, strcat('Fig/dnvn_', savename), 'png')
    saveas(gcf, strcat('Fig/dnvn_', savename), 'epsc')
end
end
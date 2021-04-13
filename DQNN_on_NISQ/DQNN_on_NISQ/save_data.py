# Plotting libs
import matplotlib.pyplot as plt # type: ignore
from matplotlib.gridspec import GridSpec # type: ignore
import tikzplotlib # type: ignore

# Math libs
import numpy as np # type: ignore
from math import ceil

# Libs for file editing
from os import mkdir, path
import glob
from json import dump
import csv

# Lib for the timestamp
from datetime import datetime

# Typing
from typing import Union, Optional, List, Tuple, Any

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# timetamp containing the date and time of execution (name of the output folder)
TIMESTAMP = "NONAME"

def make_file_structure() -> str:
    """
    Creates the output folder (if not already done) and sets the TIMESTAMP.

    Returns:
        str: Timestamp.
    """
    global TIMESTAMP
    TIMESTAMP = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    if (not path.exists("output")):
        mkdir("output")
    mkdir("output/" + TIMESTAMP) # create output folder
    
    save_execution_info(timestamp=TIMESTAMP)
        
    print('--- TIMESTAMP: {} ---'.format(TIMESTAMP))
    return TIMESTAMP
    
def save_execution_info(simulator: Optional[bool] = None,
                        network: Optional[Any] = None,
                        shots: Optional[int] = None,
                        epochs: Optional[int] = None,
                        epsilon: Optional[float] = None,
                        eta: Optional[float] = None,
                        num_training_pairs: Optional[int] = None,
                        num_validation_pairs: Optional[int] = None,
                        order_in_epsilon: Optional[int] = None,
                        gradient_method: Optional[str] = None,
                        device_name: Optional[str] = None,
                        load_params_from: Optional[str] = None,
                        load_unitary_from: Optional[str] = None,
                        load_training_states_from: Optional[str] = None,
                        load_validation_states_from: Optional[str] = None,
                        cost: Optional[float] = None,
                        validation: Optional[float] = None,
                        runtime_in_h: Optional[float] = None,
                        optimization_level: Optional[int] = None,
                        fidelity_measurement_method: Optional[str] = None,
                        timestamp: Optional[str] = None,
                        transpilation_info: Optional[str] = None,
                        gate_error_probabilities: Optional[dict] = None) -> None:
    """
    Saves the given arguments to execution_info.txt.
    """
    args = locals()
    with open('output/' + TIMESTAMP + '/execution_info.txt', "a") as f:
        for key, val in args.items():
            if val:
                f.write("{}: {}\n".format(key, val))

def save_all_params_epochs(params):
    """
    Saves the given parameters to params.txt.

    Args:
        params ([type]): A list of parameters.
    """    
    with open('output/{}/params.txt'.format(TIMESTAMP), "w") as txt_file:
        for i in range(len(params)):
            line_params = [str(x) for x in params[i][1]]
            txt_file.write("{} ".format(params[i][0]) + " ".join(line_params) + "\n")

def draw_circuit(circ: Any, filename: str='circuit.png') -> None:
    """
    Draws the quantum circuit and saves it to file called <filename>.
    The filename should end with .png.

    Args:
        circ (Any): The quantum circuit which should be drawn.
        filename (str, optional): Filename of the drawing. Defaults to 'circuit.png'.
    """
    try:
        fig = circ.draw(output='mpl', filename='output/{}/{}'.format(TIMESTAMP,filename))
        plt.close(fig)
    except:
        logger.warning('Drawing ciruit failed...')


def save(network: Optional[Any] = None, 
         all_params_epochs: Optional[Any] = None, 
         plot_list_cost: Optional[List[List[Union[Union[int,float],float]]]] = None,
         plot_list_val: Optional[List[List[Union[Union[int,float],float]]]] = None,
         plot_list_std_cost_diffs: Optional[List[List[Union[Union[int,float],float]]]] = None,
         plot_list_id: Optional[List[Union[Union[int,float],float]]] = None) -> None:
    """
    Saves and plots the given data to a file.

    Args:
        network (Optional[Any], optional): The network's class. Defaults to None.
        all_params_epochs (Optional[List[List[Union[float, List[float]]]], optional): The network's parameters per epoch. Defaults to None.
        plot_list_cost (Optional[List[List[float]]], optional): Training cost per epoch. Defaults to None.
        plot_list_val (Optional[List[List[float]]], optional): Validation cost per epoch. Defaults to None.
        plot_list_std_cost_diffs (Optional[List[List[float]]], optional): Standard deviation per epoch. Defaults to None.
        plot_list_id (Optional[List[Union[Union[int,float],float]]]): List of epoch, identity cost and (optionally) fidelity cost. Defaults to None.
    """    

    if (plot_list_id):
        with open("output/{}/identity_cost.txt".format(TIMESTAMP), "a") as f:
            f.write("{} {} {}\n".format(*plot_list_id))

    if (all_params_epochs and network): 
        save_all_params_epochs(all_params_epochs)
        plot_all_params_epochs(network, all_params_epochs)
    
    if (plot_list_cost): 
        np.savetxt("output/{}/cost.txt".format(TIMESTAMP),plot_list_cost)
        plot_cost(plot_list_cost)
        
    if (plot_list_val): 
        np.savetxt("output/{}/validation_cost.txt".format(TIMESTAMP),plot_list_val)
        plot_cost(plot_list_val, filename="validation_cost.pdf")
    
    if (plot_list_std_cost_diffs):
        np.savetxt("output/{}/standard_deviation_derivative.txt".format(TIMESTAMP),plot_list_std_cost_diffs)
        plot_standard_deviation(plot_list_std_cost_diffs)


def save_calibration_info_from_backend(backend: Any, epoch: int = 0) -> bool:
    """
    Saves the calibration info of the given backend. 
    Returns whether there was a new calibration.

    Args:
        backend (Any): The qiskit Backend.
        epoch (int, optional): Number of the epochs. Defaults to 0.

    Returns:
        (bool): Whether there was a new calibration.
    """    
    try:
        properties = backend.properties(datetime=datetime.now()).to_dict()
        basis_gates = backend.configuration().basis_gates
        
        filename = "output/{}/{}-calibration_info-0.csv".format(TIMESTAMP, properties.get("backend_name"))

        # Search last existing file
        for i in range(epoch+1):
            new_filename = "output/{}/{}-calibration_info-{}.csv".format(TIMESTAMP, properties.get("backend_name"), i)
            if (path.isfile(new_filename)):
                filename = new_filename
        
        # Dont save calibration_info if filename exists and has same last calibration date
        if (path.isfile(filename)):
            with open(filename, "r") as csv_file:
                csvreader = csv.reader(csv_file)
                next(csvreader)
                last_calibration_date = next(csvreader)[2 + len(basis_gates)]
                if last_calibration_date == str(properties.get("last_update_date")):
                    return False
        
        num_qubits = len(properties.get("qubits"))
        with open(new_filename, "w") as csv_file:
            writer = csv.writer(csv_file)
            calibration_header = ["Qubit", "Readout error"] + ["{} error".format(bg) for bg in basis_gates] + ["Date"]
            writer.writerow(calibration_header)
            for qubit in range(num_qubits):
                qubit_calibration_info: List[Union[str, int]] = [qubit]
                # single qubit readout error
                readout_error = "{:.2e}".format(properties.get("qubits")[qubit][4].get("value"))
                qubit_calibration_info.append(readout_error)
                # single qubit gate errors
                for gate in properties.get("gates", []):
                    if len(gate.get("qubits"))==1 and gate.get("qubits")[0] == qubit:
                        qubit_calibration_info.append("{:.2e}".format(gate.get("parameters")[0].get("value")))
                # two qubit gate errors
                if num_qubits > 1:
                    twoq_gate_info = ""
                    for adjacent_qubit in range(num_qubits):
                        for gate in properties.get("gates"):
                            if (len(gate.get("qubits"))==2 and gate.get("qubits")[0] == qubit and gate.get("qubits")[1] == adjacent_qubit):
                                twoq_gate_info += "{}: {:.2e}, ".format(gate.get("name"), gate.get("parameters")[0].get("value"))
                    qubit_calibration_info.append(twoq_gate_info[:-2]) # remove comma
                else: 
                    qubit_calibration_info.append("")
                qubit_calibration_info.append(properties.get("last_update_date"))
                writer.writerow(qubit_calibration_info)
        return True
    except:
        logger.warning('Not able to save the calibration info of backend device.')
        return False
    

def save_as_csv(filename: str, data: List[Any]) -> None:
    """
    Saves the data to a csv file.
    """    
    with open(filename, "a") as new_file:
        writer = csv.writer(new_file, delimiter=',')
        writer.writerow(data)


def plot_standard_deviation(plot_list: List[List[Union[Union[int,float],float]]]) -> None:
    """
    Plot the standard deviation versus the training epoch.

    Args:
        plot_list (List[List[Union[Union[int,float],float]]]): List of [epoch, standard_deviation].
    """    
    if not (isinstance(plot_list, np.ndarray)): plot_list = np.array(plot_list)
    try:
        plt.figure()
        plt.plot(plot_list[:,0], plot_list[:,1], "--o", color="b", label="Standard deviation", ms=4)
        plt.xlabel("Cost")
        plt.gca().set_ylabel(r'Standard deviation of $\frac{\partial C}{\partial \theta}$')
        # plot vertival line to mark new device calibrations
        new_calibrations = [epoch for epoch in range(len(plot_list[:,0])) if (glob.glob("output/{}/*-calibration_info-{}.csv".format(TIMESTAMP, epoch)) and epoch > 0)]
        for i, epoch in enumerate(new_calibrations):
            plt.axvline(x=plot_list[:,0][1]*epoch, color="r", linestyle="--", label="New Calibration" if i==0 else "")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/{}/standard_deviation_derivative.pdf".format(TIMESTAMP))
    finally:
        plt.close()
        
def plot_cost(plot_list: List[List[Union[Union[int,float],float]]],
              filename: str = "cost.pdf") -> None:
    """
    Plot the cost versus the learning epoch.

    Args:
        plot_list (List[List[Union[Union[int,float],float]]]): List of [epoch, cost].
        filename (str, optional): The name of the output file. Defaults to "cost.pdf".
    """     
    if not (isinstance(plot_list, np.ndarray)): plot_list = np.array(plot_list)
    try:
        plt.figure()
        plt.plot(plot_list[:,0], plot_list[:,1], "--o", color="b", label="cost", ms=4)
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        # plot device names (only if there is at least two different)
        if path.isfile("output/{}/execution_devices.txt".format(TIMESTAMP)):
            device_names = np.loadtxt("output/{}/execution_devices.txt".format(TIMESTAMP), dtype='str')
            if len(np.shape(device_names)) < 2: device_names = np.array([device_names])
            device_names = device_names[:,2]
            if not all(d == device_names[0] for d in device_names):
                ax = plt.gca()
                new_axis = ax.secondary_xaxis("top")
                new_axis.set_xticks(range(len(device_names)))
                new_axis.set_xticklabels(device_names, rotation=30, horizontalalignment='left')
            # plot vertical line to mark a new calibration
            new_calibrations = [epoch for epoch in range(len(plot_list)) if (glob.glob("output/{}/*-calibration_info-{}.csv".format(TIMESTAMP, epoch)) and epoch > 0)]
            for i, epoch in enumerate(new_calibrations):
                plt.axvline(x=epoch, color="gray", linestyle="--", label="new calibration" if i==0 else "")
                
        # plot identity cost (horizontal dashed line) for each device calibration
        if (path.isfile("output/{}/identity_cost.txt".format(TIMESTAMP))):

            id_and_fid_costs = np.loadtxt("output/{}/identity_cost.txt".format(TIMESTAMP))
            id_and_fid_costs = np.asarray([id_and_fid_costs]) if len(np.shape(id_and_fid_costs))==1 else np.asarray(id_and_fid_costs)
            # identity costs
            identity_costs = id_and_fid_costs[:,:2] 
            # fidelity costs (if given)
            fidelity_costs = id_and_fid_costs[:,::2] if all(len(c)==3 for c in id_and_fid_costs) else [] # empty if non in list
            fidelity_costs = np.asarray([cost for cost in fidelity_costs if not np.isnan(cost[1])]) # remove all np.nan
            
            # iterate through and plot the ideneity and fidelity cost
            for cf_costs, cost_name, color in zip([identity_costs, fidelity_costs],["identity cost", "fidelity cost"], ["g","r"]):
                if len(cf_costs[0]) == 0: 
                    break
                elif len(cf_costs) == 1:
                    plt.axhline(y=cf_costs[0][1], color=color, linestyle="--", label=cost_name)    
                else:
                    plt.plot(cf_costs[:,0], cf_costs[:,1], color=color, linestyle="--", label=cost_name)

        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/{}/{}".format(TIMESTAMP, filename))
    except:
        logger.warning("Cost could not be plotted. An error occured.")
    finally:
        plt.close()
        
def plot_all_params_epochs_diff(timestamp1: str,
                                timestamp2: str) -> None:
    """
    Takes two folder names (timestamps) and plots both parameters in a single plot.
    The first data set is plotted with a solid line, the second one with a dashed line.
    The plot is saved in both folders.

    Args:
        timestamp1 (str): Timestamp of first data folder.
        timestamp2 (str): Timestamp of second data folder.
    """
    try:
        filename1 = "output/{}/params.txt".format(timestamp1)
        filename2 = "output/{}/params.txt".format(timestamp2)
        data1 = np.loadtxt(filename1)
        data2 = np.loadtxt(filename2)
        data1 = [[d[0],d[1:]] for d in data1]
        data2 = [[d[0],d[1:]] for d in data2]   
        all_params_epochs_list = [data1, data2]
        try:
            plt.figure()
            prev_plot_colors: List[str] = []
            for k, all_params_epochs in enumerate(all_params_epochs_list):
                epoch_list = [param[0] for param in all_params_epochs]
                params = [param[1] for param in all_params_epochs]
                param_per_epoch = [[params[i][j] for i in range(len(params))] for j in range(len(params[0]))]
                for e, param in enumerate(param_per_epoch):
                    if k > 0:
                        plt.plot(epoch_list, param, "-"*(k+1), c=prev_plot_colors[e])
                    else:
                        p = plt.plot(epoch_list, param, "-")
                        prev_plot_colors.append(p[0].get_color())
            plt.xlabel("Epoch")
            plt.ylabel("Parameter values")
            plt.savefig("output/{}/param_tracker-comparison_to_{}.pdf".format(timestamp1, timestamp2))
            plt.savefig("output/{}/param_tracker-comparison_to_{}.pdf".format(timestamp2, timestamp1))
        finally:
            plt.close()
    except:
        if (not path.isfile("output/{}/params.txt".format(timestamp1))):
            logger.warning('The following file does not exist:\noutput/{}/params.txt'.format(timestamp1))
        if (not path.isfile("output/{}/params.txt".format(timestamp2))):
            logger.warning('The following file does not exist:\noutput/{}/params.txt'.format(timestamp2))


def plot_all_params_epochs(network: Any,
                           all_params_epochs: List[List[Any]]) -> None:
    """
    Plots the values of all network parmaters versus the training epoch.
    Plot is saved as params.pdf.
    
    [[{title: "Layer 1, Preparation", params: [params per epoch]}, ... (neuronwise)], ... (layerwise)]

    Layerwise
    ----------------
    |       |       | Neuronwise  
    -----------------
    |       |       |
    -----------------

    Args:
        network (Any): The network's class.
        all_params_epochs (List[List[Any]]): List of [epoch, network parameters].
    """    
    try:
        epoch_list = [param[0] for param in all_params_epochs]
        param_list = [param[1] for param in all_params_epochs]
        params_per_epoch: List[List[float]] = [[param_list[i][j] for i in range(len(param_list))] for j in range(len(param_list[0]))]
        reshaped_params: List[List[dict]] = network.reshape_params_for_plotting(params_per_epoch)
        fig = plt.figure(tight_layout=True)
        gs = GridSpec(nrows=len(reshaped_params), ncols=np.prod([len(x) for x in reshaped_params]))
        for l in range(len(reshaped_params)):
            gs_col_size = np.prod([len(x) for x in reshaped_params])//len(reshaped_params[l])
            for j in range(len(reshaped_params[l])):
                new_ax = fig.add_subplot(gs[l,j*gs_col_size:(j+1)*gs_col_size])
                new_ax.set_xlabel("Epoch")
                new_ax.set_ylabel("Parameter values")
                new_ax.set_title(reshaped_params[l][j]["title"])
                for k, param in enumerate(reshaped_params[l][j]["params"]):
                    new_ax.plot(epoch_list, param, label="Param #{}".format(k+1))
        fig.savefig('output/{}/params.pdf'.format(TIMESTAMP))
    finally:
        plt.close()

def plot_cost_mean(timestamp_start: str, 
                   timestamp_end: str, 
                   cost_filename: str = "cost", 
                   plot_single_costs: bool = False,
                   optimization_level: Optional[int] = None,
                   num_training_pairs: Optional[int] = None) -> None:
    """
    Plots the mean of different costs (which cost is specified by the filename).
    All files are used starting from timestamp_start to timestamp_end.
    The files can be filtered by optimization level and the number of training pairs.
    The plot is saved in the output folder.

    Args:
        timestamp_start (str): Folder (timestamp) of the first execution.
        timestamp_end (str): Folder (timestamp) of the last execution.
        cost_filename (str, optional): The filename of the respective cost (the cost's txt file). Defaults to "cost".
        plot_single_costs (bool, optional): Whether the single costs should be plotted versus the epoch (or only the overall mean). Defaults to False.
        optimization_level (Optional[int], optional): The optimization level (used for filtering the cost files). Defaults to None.
        num_training_pairs (Optional[int], optional): The number of training pairs (used for filtering the cost files). Defaults to None.
    """    
    try:
        folders = glob.glob("output/*/") # all folder names (timestamps) of the output folder
        costs = []
        identity_costs = []
        for folder in folders:
            if folder >= "output/{}/".format(timestamp_start) and folder <= "output/{}/".format(timestamp_end):
                # timestamp is in range [timestamp_start, timestamp_end]
                try:
                    with open("{}execution_info.txt".format(folder), "r") as f:
                        content = f.readlines()
                        opt_level = optimization_level
                        for c in content:
                            if c.split(":")[0] == "num_training_pairs":
                                num_tp = int(c.split(":")[1].strip()) # number of training pairs of this execution
                            if c.split(":")[0] == "optimization_level":
                                opt_level = int(c.split(":")[1].strip()) # optimization level of this execution
                        if (opt_level == optimization_level or optimization_level is None) and (num_tp == num_training_pairs or num_training_pairs is None):
                            # load and append cost to list if optimization level and number of training pairs match the requirements
                            cost = np.asarray(np.loadtxt("{}{}.txt".format(folder, cost_filename)))
                            identity_cost = np.asarray(np.loadtxt("{}identity_cost.txt".format(folder)))
                            costs.append(cost)
                            identity_costs.append(np.mean(identity_cost[:,1]))
                except:
                    logger.warning("Could not load {} or {}.".format("{}{}.txt".format(folder, cost_filename), "{}identity_cost.txt".format(folder)))
        
        if plot_single_costs:
            # To plot all the costs which are given
            min_epochs = int(min([cost[-1][0] for cost in costs]))+1
            max_epochs = int(max([cost[-1][0] for cost in costs]))+1
            for i in range(len(costs)):
                puffer = np.array([[np.nan,np.nan]]*(max_epochs-len(costs[i])))
                if len(puffer) > 0:
                    costs[i] = np.concatenate((costs[i], puffer), axis=0)
            cost_list = [cost[:,1] for cost in costs]
            costs_mean = np.mean(cost_list, axis=0)
            costs_std = np.std(cost_list, axis=0)
            id_costs_mean = np.full_like(costs_mean, np.nanmean(identity_costs))
            id_costs_std = np.full_like(costs_mean, np.nanstd(identity_costs))
        else:
            # Plot only for epochs where every file contains a cost
            min_epochs = int(min([cost[-1][0] for cost in costs]))+1
            costs = [cost[:min_epochs] for cost in costs]
            cost_list = [cost[:,1] for cost in costs]
            costs_mean = np.mean(cost_list, axis=0)
            costs_std = np.std(cost_list, axis=0)
            id_costs_mean = np.asarray([np.nanmean(identity_costs)]*min_epochs)
            id_costs_std = np.asarray([np.nanstd(identity_costs)]*min_epochs)

        plt.title(cost_filename)

        # Plot mean cost with std
        epoch_step = costs[0][1][0]
        epoch_list = np.arange(0, min_epochs, epoch_step)
        for cost in costs:
            plt.plot(epoch_list, cost[:,1], "--o", color="k", ms=2, lw=0.9)
        plt.plot(epoch_list, costs_mean, "--o", color="b", ms=4)
        plt.fill_between(epoch_list, costs_mean-costs_std, costs_mean+costs_std, alpha=0.25, color="b")

        # Plot mean identity cost with std
        plt.plot(id_costs_mean, "--", color="g", ms=4)
        plt.fill_between(range(len(id_costs_mean)), id_costs_mean-id_costs_std, id_costs_mean+id_costs_std, alpha=0.5, color="g")
        
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.savefig("output/mean_{}{}{}.pdf".format(cost_filename, "-optlvl_{}".format(optimization_level) if not optimization_level is None else "", "-{}_tps".format(num_training_pairs or "")))
    finally:
        plt.close()


def plot_cost_comp(timestamp_start: str, 
                   timestamp_end: str, 
                   num_training_pairs: int, 
                   cost_filename: str = "cost") -> None:
    """
    Plots all costs in a given range (timestamp_start to timestamp_end) versus the epoch.
    The costs can be filtered by the number of training pairs.
    The plot is saved in the output folder.

    Args:
        timestamp_start (str): The folder (timestamp) of the first execution.
        timestamp_end (str): The folder (timestamp) of the last execution.
        num_training_pairs (int): The number of training pairs (used for filtering the costs).
        cost_filename (str, optional): The filename of the cost (the cost's txt file). Defaults to "cost".
    """    
    try:
        folders = glob.glob("output/*/")
        costs = []
        identity_costs = []
        hyper_parameters = []
        timestamps = []
        for folder in folders:
            if folder >= "output/{}/".format(timestamp_start) and folder <= "output/{}/".format(timestamp_end):
                with open("{}execution_info.txt".format(folder), "r") as f:
                    content = f.readlines()
                    for c in content:
                        if c.split(":")[0] == "epsilon":
                            eps = c.split(":")[1].strip()
                        if c.split(":")[0] == "eta":
                            eta = c.split(":")[1].strip()
                        if c.split(":")[0] == "num_training_pairs":
                            do_plot = int(c.split(":")[1].strip()) == num_training_pairs
                    if do_plot and eta and eps:
                        try:
                            cost = np.asarray(np.loadtxt("{}{}.txt".format(folder, cost_filename)))
                            identity_cost = np.asarray(np.loadtxt("{}identity_cost.txt".format(folder)))
                            costs.append(cost)
                            identity_costs.append(identity_cost)
                            hyper_parameters.append([eps, eta])
                            timestamps.append(folder.split("output/")[1][:-1])
                        except:
                            logger.warning("Could not load {} or {}.".format("{}{}.txt".format(folder, cost_filename), "{}identity_cost.txt".format(folder)))
        cmap = plt.get_cmap('jet')
        plot_colors = cmap(np.linspace(0, 1.0, len(costs)))
        plt.figure(figsize=(8,5))
        for i, cost in enumerate(costs):
            p = plt.plot(cost[:,0], cost[:,1], "-o", color=plot_colors[i], label="eps = {}, eta = {}\n{}".format(*hyper_parameters[i], timestamps[i]), ms=1)
        for i, identity_cost in enumerate(identity_costs):
            for k, c in enumerate(identity_cost):
                plt.plot([c[0],identity_cost[k+1][0] if k+1 < len(identity_cost) else costs[i][-1,0]],[c[1],c[1]], color=plot_colors[i], linestyle="--")
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Number of training pairs: {}'.format(num_training_pairs))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.savefig("output/{}-comp-{}_tps.pdf".format(cost_filename, num_training_pairs), bbox_inches='tight')
    finally:
        plt.close()


def plot_cost_contour(timestamp_start: str, 
                      timestamp_end: str, 
                      num_training_pairs: int,
                      cost_filename: str = "cost",
                      avg_len: int = 1,
                      cost_ratio: bool = False) -> None:
    """
    Creates a contour plot of the cost for each execution between timestamp_start and timestamp_end.
    The cost is plotted over epsilon (horizontal axis) and eta (vertical axis).
    The costs can be filtered by the number of training pairs.
    The plot is saved in the output folder.

    Args:
        timestamp_start (str): The folder (timestamp) of the first execution.
        timestamp_end (str): The folder (timestamp) of the last execution.
        num_training_pairs (int): The number of training pairs (used for filtering the costs).
        cost_filename (str, optional): The filename of the cost (the cost's txt file). Defaults to "cost".
        avg_len (int, optional): Number of epochs over which the cost should be averaged (starting from the last cost). Defaults to 1.
        cost_ratio (bool, optional): Whether the ratio between the cost and the identity cost should be plotted. Defaults to False.
    """    
    
    folders = glob.glob("output/*/")
    eps_list = []
    eta_list = []
    costs = []
    for folder in folders:
        if folder >= "output/{}/".format(timestamp_start) and folder <= "output/{}/".format(timestamp_end):
            with open("{}execution_info.txt".format(folder), "r") as f:
                content = f.readlines()
                for c in content:
                    if c.split(":")[0] == "epsilon":
                        eps = float(c.split(":")[1].strip())
                    if c.split(":")[0] == "eta":
                        eta = float(c.split(":")[1].strip())
                    if c.split(":")[0] == "num_training_pairs":
                        do_plot = int(c.split(":")[1].strip()) == num_training_pairs
                if do_plot and eta and eps:
                    try:
                        # Load cost data and set plot value accordingly
                        cost = np.asarray(np.loadtxt("{}{}.txt".format(folder, cost_filename)))
                        # Load identity cost
                        identity_cost = np.asarray(np.loadtxt("{}identity_cost.txt".format(folder)))
                        # Add epsilon and eta to lists
                        eps_list.append(eps)
                        eta_list.append(eta)
                        step = cost[-1,0] - cost[-2,0] if len(cost) > 1 else 1
                        cost = np.average(cost[:,1][-max(1, avg_len//step):]) # Average over the last avg_len epochs
                        cost = cost/np.average(identity_cost[:,1]) if cost_ratio else cost # Take the ratio if cost_ratio == True
                        # Add cost to list
                        costs.append(cost)
                    except:
                        logger.warning("Could not load {} or {}.".format("{}{}.txt".format(folder, cost_filename), "{}identity_cost.txt".format(folder)))
    try:
        fig, ax = plt.subplots(1,1)

        plt_eps_list = [] 
        plt_eta_list = [] 
        plt_cost_list = []
        
        plt_eps_list = sorted(list(set(eps_list)))
        plt_eta_list = sorted(list(set(eta_list)))

        # finds the costs for the respective epsilon and eta
        for eps in plt_eps_list:
            eta_cost = []
            for eta in plt_eta_list:
                eps_i = [i for i,e in enumerate(eps_list) if e == eps]
                eta_i = [i for i,e in enumerate(eta_list) if e == eta]
                matching_i = list(set(eps_i).intersection(eta_i))
                cost = np.mean([costs[i] for i in matching_i])
                eta_cost.append(cost)
            plt_cost_list.append(eta_cost)
            
        cp = ax.pcolormesh(plt_eta_list, plt_eps_list, plt_cost_list, shading='auto', cmap=plt.get_cmap("RdYlGn"))
        fig.colorbar(cp, ax=ax)
        ax.set_xlabel('Eta')
        ax.set_ylabel('Epsilon')
        plt.title('Training pairs: {}, {}'.format(num_training_pairs, 'Ratio {}/identity cost'.format(cost_filename) if cost_ratio else cost_filename))
        plt.tight_layout()
        plt.savefig("output/{}-contour-{}_tps.pdf".format(cost_filename, num_training_pairs))
    finally:
        plt.close()


def plot_cost_vs_error_probability(csv_filename: str = "test_different_error_probabilities.csv",
                                   csv_error_prob_col: int = 2,
                                   cost_filename: str = "cost",
                                   avg_len: int = 10):
    """
    Plots the average costs versus the error probability factor.
    The cost_filename gives the cost type (training or validation cost).

    Args:
        csv_filename (str, optional): The csv filename containing all the relevant folders (timestamps). Defaults to "test_different_error_probabilities.csv".
        csv_error_prob_col (int, optional): The column number of the csv where the error probability factor is stored. Defaults to 2.
        cost_filename (str, optional): The filename of the cost (the cost's txt file). Defaults to "cost".
        avg_len (int, optional): Number of epochs over which the cost should be averaged (starting from the last cost). Defaults to 10.
    """    
    timestamps = []
    error_prob_facs = []
    with open("output/{}".format(csv_filename), "r") as csv_file:
        csvreader = csv.reader(csv_file)
        next(csvreader)
        for row in csvreader:
            timestamps.append(row[0])
            error_prob_facs.append(float(row[csv_error_prob_col]))
    epf_sorted_unique = sorted(list(set(error_prob_facs)))

    folders = glob.glob("output/*/")
    costs: List[List[Any]] = [[[] for _ in range(len(epf_sorted_unique))] for _ in range(2)] # dqnn, qaoa
    id_costs: List[List[Any]] = [[[] for _ in range(len(epf_sorted_unique))] for _ in range(2)] # dqnn, qaoa
    for t, timestamp in enumerate(timestamps):
        with open("output/{}/execution_info.txt".format(timestamp), "r") as f:
            content = f.readlines()
            for c in content:
                if c.split(":")[0] == "network":
                    network_i = int("QAOA" in c.split(":")[1])
            epf_i = epf_sorted_unique.index(error_prob_facs[t])
            cost = np.average(np.asarray(np.loadtxt("output/{}/{}.txt".format(timestamp, cost_filename)))[:,1][-avg_len:])
            costs[network_i][epf_i].append(cost)
            id_cost = np.average(np.asarray(np.loadtxt("output/{}/identity_cost.txt".format(timestamp)))[:,1])
            id_costs[network_i][epf_i].append(id_cost)

    costs_mean = [[np.mean(costs[i][j]) if costs[i][j] else np.nan for j in range(len(epf_sorted_unique))] for i in range(2)]
    costs_std = [[np.std(costs[i][j]) if costs[i][j] else np.nan for j in range(len(epf_sorted_unique))] for i in range(2)]
    id_costs_mean = [[np.mean(id_costs[i][j]) if id_costs[i][j] else np.nan for j in range(len(epf_sorted_unique))] for i in range(2)]
    try: 
        plot_colors = ["#785EF0", "#DC267F"]
        labels = ["DQNN", "QAOA"]
        for n in range(2):
            plt.errorbar(epf_sorted_unique, costs_mean[n], yerr=costs_std[n], marker="o", linestyle="--", color=plot_colors[n], label=labels[n], capsize=3)
            plt.plot(epf_sorted_unique, id_costs_mean[n], "-.", color=plot_colors[n], label="Identity cost ({})".format(labels[n]))
        plt.xlabel("Error probability factor")
        plt.ylabel(cost_filename)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("output/{}_vs_error_prob.pdf".format(cost_filename))
    finally: 
        plt.close()


def generate_tikz(figure: Any, filename: str):
    """
    Generates the LATEX (tikz) code of a given figure. 
    The code is stored in output/tikz_figures/<filename>.tex

    Args:
        figure (Any): The figure which shuold be converted to tex.
        filename (str): The name of the tex file.
    """    
    if not path.isdir('output/tikz_figures'):
        mkdir('output/tikz_figures')
    tikzplotlib.save("output/tikz_figures/{}.tex",format(filename), figure=figure, strict=True, wrap=True)


if __name__ == "__main__":
    # python save_data.py
    
    logger.info('Running save_data.py as main...\n')
    
    start_timestamp = "2021_01_04-22_40_07"
    end_timestamp = "2021_01_05-09_16_43"
    # training_pairs = 4
    # plot_cost_mean(start_timestamp, end_timestamp, num_training_pairs=4)
    # for tp in range(1,5):
    #     for cost_filename in ["cost", "validation_cost"]:
    #         # plot_cost_comp(start_timestamp, end_timestamp, training_pairs, cost_filename)
    #         # plot_cost_contour(start_timestamp,end_timestamp, training_pairs, cost_filename)
    #         plot_cost_mean(start_timestamp, end_timestamp, cost_filename, num_training_pairs=tp)
    
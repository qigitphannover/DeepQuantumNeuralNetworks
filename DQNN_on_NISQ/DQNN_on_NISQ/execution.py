import training
import save_data as sd
from network_classes import Network_DQNN
from network_classes import Network_QAOA
import network_classes as nc
from training_data import generate_training_data

import numpy as np # type: ignore

from typing import List, Optional, Union, NamedTuple, Any

def single_execution(simulator: bool = True,
                        network: Union[Network_DQNN, Network_QAOA] = Network_DQNN(qnn_arch=[2,2]),
                        shots: int = 2**13,
                        epochs: int = 100,
                        epsilon: float = 0.25,
                        eta: float = 0.5,
                        order_in_epsilon: int = 2,
                        gradient_method: str = 'gradient_descent',
                        optimization_level: int = 3,
                        num_training_pairs: Optional[int] = None,
                        num_validation_pairs: Optional[int] = None,
                        device_name: Optional[str] = None,
                        validation_step: Optional[int] = None,
                        load_params_from: Optional[str] = None,
                        load_unitary_from: Optional[str] = None,
                        load_training_states_from: Optional[str] = None,
                        load_validation_states_from: Optional[str] = None,
                        gate_error_probabilities: Optional[dict] = None) -> Any:
    """
    User-level function to set up a network and train it on a given device.
    Default network is the DQNN from "Training Quantum Neural Networks on NISQ devices" on a noise-free simulator.
    Creates a results folder in the output directory named by a timestamp. Saves all relevant parameters. Creates relevant figures.

    Args:
        simulator (bool, optional): If True, the network is trained using a simulator instead of a real quantum device.
            It depends on the device_name if the simulator imitates the noise and qubit coupling of a real device
            or executes the corresponding quantum circuits noise-free.
            Defaults to True.
        network (Union[Network_DQNN, Network_QAOA], optional): A network object which should be trained.
            The object is defined by the class of network, its architecture and for QAOA the hermitian matrices A,B.
            Defaults to Network_DQNN(qnn_arch=[2,2]).
        shots (int, optional): Number of shots per circuit evaluation.
            Defaults to 2**13.
        epochs (int, optional): Numer of epochs for training the given network.
            Defaults to 100.
        epsilon (float, optional): Hyperparameter for optimization method. Step size for finite gradient calculation.
            Defaults to 0.25.
        eta (float, optional): Hyperparameter for optimization method. Factor to the gradient when calculating new parameters.
            Defaults to 0.5.
        order_in_epsilon (int, optional): Order of finite gradient calculation. Possible values are 1 and 2.
            Defaults to 2.
        gradient_method (str, optional): Optimization method whereby conjugate gradient and gradient descent are possible.
            Conjugate gradient is not stable.
            Defaults to 'gradient_descent'.
        optimization_level (int, optional): Level with which the circuit is optimized by qiskit. Higher values mean better optimization,
            e.g. in terms of fewer 2-qubit gates and noise-adjusted qubit layout.
            Defaults to 3.
        num_training_pairs (Optional[int], optional): Number of training pairs.
            If no value is given a determining number of training pairs is used to learn the unitary.
            Defaults to None.
        num_validation_pairs (Optional[int], optional): Number of validation pairs.
            If no value is given a determining number of validation pairs is used to validate the network unitary.
            Defaults to None.
        device_name (Optional[str], optional): The device name of a IBMQ device.
            If a device is given either the noise properties and the qubit layout of this device are simulated (simulator==True)
            or the network is trained on the real quantum device (simulator==False). Dependend on the IBMQ user account a long queuing time should be expected.
            The device must be accessible for the IBMQ user. The IBMQ user account can be configured in user_config.py. 
            If no device name is given a noise-free simulator is used (simulator==True) or the least busy device is used (simulator==False).
            Defaults to None.
        validation_step (Optional[int], optional): Determines the frequency in which the network's learning is validated using unknown validation pairs.
            If no value is given no validation is executed.
            Defaults to None.
        load_params_from (Optional[str], optional): The network can be initialized with specific parameters which can be loaded from a local txt file.
            If no file is given the initial parameters are generated randomly.
            Defaults to None.
        load_unitary_from (Optional[str], optional): The target unitary can be loaded from a local txt file.
            If no file is given the target unitary is generated randomly.
            Defaults to None.
        load_training_states_from (Optional[str], optional): The network can be trained with specific training pairs which can be loaded from a local txt file.
            If no file is given the training pairs are generated randomly.
            Defaults to None.
        load_validation_states_from (Optional[str], optional): The network can be validated with specific validation pairs which can be loaded from a local txt file.
            If no file is given the validation pairs are generated randomly.
            Defaults to None.
        gate_error_probabilities (Optional[dict], optional): To simulate different gate noise levels of a quantum device its gate error probablities can be given here.
            These probabilties are used to create a depolarized quantum error channel and its corresponding simulator. Here, the qubit layout of ibmq_16_melbourne is used.
            If gate error probalities are given, the device_name and the simulator arguments are ignored.
            Defaults to None.

    Returns:
        Any: A tuple of the following properties (used in experiments with shared properties, e.g. parameters and unitary)
            1. initial parameters (List[float])
            2. target unitary (qt.Qobj)
            3. input states of training pairs (List[qt.Qobj])
            4. input states of validation pairs (List[qt.Qobj])
            5. Hermitian operator A if a QAOA is used (qt.Qobj)
            6. Hermitian operator B if a QAOA is used (qt.Qobj)
    """
    # Save timestamp and create results folder
    timestamp = sd.make_file_structure()
    
    network.gate_error_probabilities = gate_error_probabilities or {}
    
    network.update_params(load_params_from)
    initial_params = network.params

    # Generate or load training data, i.e., training pairs, validation pairs and the target unitary
    training_pairs, validation_pairs, network_unitary = generate_training_data(network.qnn_arch[0], 
        num_training_pairs, num_validation_pairs, network_unitary=load_unitary_from, 
        input_training_states=load_training_states_from, input_validation_states=load_validation_states_from)
    
    # Fix device
    if not simulator:
        device_name = training.get_device(network, simulator, device_name=device_name, do_calibration=False).name()

    nc.construct_noise_model(network)

    # Pre-transpile parametrized circuits
    nc.construct_and_transpile_circuits(network, training_pairs, validation_pairs, optimization_level, device_name, simulator)
    
    # Save all relevant parameters to an execution_info.txt
    sd.save_execution_info(simulator, network, shots, epochs, epsilon, eta, len(training_pairs), 
        len(validation_pairs), order_in_epsilon, gradient_method, device_name, 
        load_params_from if isinstance(load_params_from, str) else None, load_unitary_from if isinstance(load_unitary_from, str) else None, 
        optimization_level=optimization_level, fidelity_measurement_method=network.fid_meas_method,
        gate_error_probabilities=gate_error_probabilities or None)
    
    # Save relevant data to the results folder
    np.savetxt("output/{}/input_training_states.txt".format(timestamp), [tp[0] for tp in training_pairs])
    np.savetxt("output/{}/input_validation_states.txt".format(timestamp), [vp[0] for vp in validation_pairs])
    np.savetxt("output/{}/unitary.txt".format(timestamp), network_unitary)
    if (isinstance(network, Network_QAOA)):
        np.savetxt("output/{}/matrix_A.txt".format(timestamp), network.A)
        np.savetxt("output/{}/matrix_B.txt".format(timestamp), network.B)

    # Train the network
    trained_params, _ = training.train_network(network, training_pairs, 
        epochs=epochs, eta=eta, simulator=simulator, shots=shots, epsilon=epsilon, 
        order_in_epsilon=order_in_epsilon, gradient_method=gradient_method, device_name=str(device_name), 
        validation_step=validation_step, validation_pairs=validation_pairs)
    print('--- TIMESTAMP: {} ---'.format(timestamp))

    return initial_params, network_unitary, [tp[0] for tp in training_pairs], [vp[0] for vp in validation_pairs], network.A if isinstance(network, Network_QAOA) else None, network.B if isinstance(network, Network_QAOA) else None
                
def train_for_different_start_parameters(num_runs: int = 1, num_training_pairs: List[int] = [4]):
    """
    Helper function used for generalization analysis in figure 2a in "Training Quantum Neural Networks on NISQ devices".
    Uses different unitaries, start parameters and training/validation pairs
    for each full training run (all different numbers of training pairs)
    while using the same unitary and training/validation pairs for DQNN and QAOA.

    Args:
        num_runs (int, optional): Number of full runs. Defaults to 1.
        num_training_pairs (List[int], optional): List of number of training pairs for which both networks should be trained.
            Defaults to [4].
    """    
    csv_filename = "output/test_different_init_params.csv"
    sd.save_as_csv(csv_filename, ["timestamp", "network type", "number of tps", "epochs", "cost", "validation cost", "ratio_cost_to_cal"])
    for i in range(num_runs):
        training.DEVICE = None
        unitary, training_pairs, validation_pairs = None, None, None
        training_pairs_save, validation_pairs_save = None, None
        for network_type in ["dqnn", "qaoa"]:
            initial_params, A, B = None, None, None
            if training_pairs_save:
                training_pairs, validation_pairs = training_pairs_save.copy(), validation_pairs_save.copy()
            for j, tp in enumerate(sorted(num_training_pairs, reverse=True)):
                print('\n\n\nStart run {} of {}\n'.format(i*len(num_training_pairs)+j+1, num_runs*len(num_training_pairs)))
                initial_params, unitary, training_pairs, validation_pairs, A, B = single_execution(
                    simulator=True,
                    network=Network_DQNN(qnn_arch=[2,2], fidelity_measurement_method='destructive_swap') if network_type == "dqnn" \
                        else Network_QAOA(num_qubits=2, num_params_fac=1, fidelity_measurement_method='destructive_swap', A=A, B=B),
                    shots=2**13,
                    epochs=1000 if network_type == 'dqnn' else 1000,
                    epsilon=0.5 if network_type == "dqnn" else 0.15,
                    eta=1.0 if network_type == "dqnn" else 0.1,
                    order_in_epsilon=2,
                    gradient_method='gradient_descent',
                    validation_step=5,
                    num_training_pairs=tp,
                    load_unitary_from=unitary,
                    load_training_states_from=training_pairs,
                    load_validation_states_from=validation_pairs,
                    load_params_from=initial_params,
                    device_name='ibmq_casablanca',
                    optimization_level=3
                )
                if not training_pairs_save:
                    training_pairs_save, validation_pairs_save = training_pairs.copy(), validation_pairs.copy()
                epochs, cost = np.loadtxt('output/{}/cost.txt'.format(sd.TIMESTAMP))[-1]
                val_cost = np.loadtxt('output/{}/validation_cost.txt'.format(sd.TIMESTAMP))[-1][1]
                identity_cost_avg = np.average(np.loadtxt('output/{}/identity_cost.txt'.format(sd.TIMESTAMP))[:,1])
                cost_id_ratio = cost/identity_cost_avg
                data = [sd.TIMESTAMP, network_type, tp, epochs, cost, val_cost, cost_id_ratio]
                sd.save_as_csv(csv_filename, data)


def train_for_different_error_probabilities(num_runs: int = 1):
    """
    Helper function used for gate noise analysis in figure 2b in "Training Quantum Neural Networks on NISQ devices".
    Uses different unitaries, start parameters and training/validation pairs
    for each full training run (all different error probabilities)
    while using the same unitary and training/validation pairs for DQNN and QAOA.
    
    Args:
        num_runs (int, optional): Number of full runs. Defaults to 1.
    """    

    csv_filename = "output/test_different_error_probabilities.csv"
    sd.save_as_csv(csv_filename, ["timestamp", "network type", "error_fac", "epochs", "cost", "validation cost", "ratio_cost_to_cal"])
    error_fac_list = sorted([4, 2, 1, 0.5, 0.25, 0])
    for j in range(num_runs):
        unitary, training_pairs, validation_pairs = None, None, None
        training.DEVICE = None
        for network_type in ['dqnn', 'qaoa']:
            initial_params, A, B = None, None, None
            i=0
            while i < len(error_fac_list):
                error_fac = error_fac_list[i]
                print('\n\n\nStart run {} of {}\n'.format(len(error_fac_list)*j+i+1, len(error_fac_list)*5))
                initial_params, unitary, training_pairs, validation_pairs, A, B = single_execution(
                    simulator=True,
                    network=Network_DQNN(qnn_arch=[2,2], fidelity_measurement_method='destructive_swap') if network_type == "dqnn" \
                        else Network_QAOA(num_qubits=2, num_params_fac=1, fidelity_measurement_method='destructive_swap', A=A, B=B),
                    shots=2**13,
                    epochs=600 if network_type == "dqnn" else 800,
                    epsilon=0.25 if network_type == "dqnn" else 0.05,
                    eta=0.5 if network_type == "dqnn" else 0.05,
                    order_in_epsilon=2,
                    gradient_method='gradient_descent',
                    validation_step=5,
                    load_unitary_from=unitary,
                    load_training_states_from=training_pairs,
                    load_validation_states_from=validation_pairs,
                    load_params_from=initial_params,
                    optimization_level=3,
                    # gate_error_probabilities={'u2': [3.e-4*error_fac,1],'u3': [6.e-4*error_fac,1],'cx': [1.e-2*error_fac,2]} # ibmq_casablanca
                    gate_error_probabilities={'sx': [1.179e-3*error_fac,1],'x': [1.179e-3*error_fac,1],'cx': [3.142e-2*error_fac,2]} # ibmq_melbourne
                )
                epochs, cost = np.loadtxt('output/{}/cost.txt'.format(sd.TIMESTAMP))[-1]
                val_cost = np.loadtxt('output/{}/validation_cost.txt'.format(sd.TIMESTAMP))[-1][1]
                identity_cost_avg = np.average(np.loadtxt('output/{}/identity_cost.txt'.format(sd.TIMESTAMP))[:,1])
                if (i == 0 and val_cost > 0.95 * identity_cost_avg) or i>0:
                    cost_id_ratio = cost/identity_cost_avg
                    data = [sd.TIMESTAMP, network_type, error_fac, epochs, cost, val_cost, cost_id_ratio]
                    sd.save_as_csv(csv_filename, data)
                    i+=1
                else:
                    initial_params=None

def train_for_different_hyperparams(eta_list: List[float], epsilon_list: List[float]) -> None:
    """
    Helper function to train the network for different hyper parameters to find the most efficient.

    Args:
        eta_list (List[float]): List of eta values to be tested.
        epsilon_list (List[float]): List of epsilon values to be tested.
    """    
    
    # folder = "input/qaoa_2qubits"
    folder = "input/dqnn_2-2"
    
    csv_filename = "output/test_different_hyperparams_qaoa.csv"
    sd.save_as_csv(csv_filename, ["timestamp", "eta", "epsilon", "training pairs", "epochs", "cost", "validation cost", "ratio_cost_to_cal"])
    for eta in eta_list:
        for epsilon in epsilon_list:
            num_training_pairs = 4
            while num_training_pairs > 0:
                print('\n\n\nStart run for epsilon={}, eta={}\n'.format(epsilon, eta))
                single_execution(
                    simulator=True,
                    # network=Network_QAOA(num_qubits=2, num_params_fac=1, fidelity_measurement_method='destructive_swap', A=folder+'/matrix_A.txt', B=folder+'/matrix_B.txt'),
                    network=Network_DQNN(qnn_arch=[2,2], fidelity_measurement_method='destructive_swap'),
                    shots=2**14,
                    epochs=1000,
                    epsilon=epsilon,
                    eta=eta,
                    order_in_epsilon=2,
                    gradient_method='gradient_descent',
                    validation_step=5,
                    num_training_pairs=num_training_pairs,
                    load_unitary_from=folder+'/unitary.txt',
                    load_params_from=folder+'/params.txt',
                    load_training_states_from=folder+'/input_training_states.txt',
                    load_validation_states_from=folder+'/input_validation_states.txt',
                    device_name='ibmq_casablanca',
                    optimization_level=3
                )
                epochs, cost = np.loadtxt('output/{}/cost.txt'.format(sd.TIMESTAMP))[-1]
                val_cost = np.loadtxt('output/{}/validation_cost.txt'.format(sd.TIMESTAMP))[-1][1]
                identity_cost_avg = np.average(np.loadtxt('output/{}/identity_cost.txt'.format(sd.TIMESTAMP))[:,1])
                cost_id_ratio = cost/identity_cost_avg
                data = [sd.TIMESTAMP, eta, epsilon, num_training_pairs, epochs, cost, val_cost, cost_id_ratio]
                sd.save_as_csv(csv_filename, data)
                print('\nFinished run for epsilon={}, eta={}'.format(epsilon, eta))
                num_training_pairs = 0 if (cost_id_ratio < 0.9 and num_training_pairs == 4) else num_training_pairs-1

import save_data as sd

import scipy as sc # type: ignore
import numpy as np # type: ignore
from typing import Union, Optional, List, Tuple, Any

def randomQubitUnitary(numQubits):
    """
    returns a unitary  2^(numQubits)×2^(numQubits)-matrix
    as a numpy array (np.ndarray) that is the tensor product
    of numQubits factors. 
    Before orthogonalization, it's elements are randomly picked
    out of a normal distribution.
    """
    dim = 2**numQubits
    #Make unitary matrix
    res = sc.random.normal(size=(dim,dim)) + 1j * sc.random.normal(size=(dim,dim))
    res = sc.linalg.orth(res)
    #Return
    return res

def randomQubitState(numQubits):
    dim = 2**numQubits
    #Make normalized state
    res = sc.random.normal(size=(dim,1)) + 1j * sc.random.normal(size=(dim,1))
    res = (1/sc.linalg.norm(res)) * res
    #Return
    return res.flatten()

def generate_training_data(num_qubits: int,
                           num_training_pairs: Optional[int] = None,
                           num_validation_pairs: Optional[int] = None,
                           network_unitary: Optional[Union[np.ndarray,List[complex],str]] = None,
                           input_training_states: Optional[Union[List[np.ndarray],str]] = None,
                           input_validation_states: Optional[Union[List[np.ndarray],str]] = None) -> Tuple[
                                                                        List[List[np.ndarray]],
                                                                        List[List[np.ndarray]], np.ndarray]:    
    """
    generate_training_data is used to prepare a given number of training pairs
    for a given network architecture such that one training pair can be used
    to directly initialize the first 2m qubits of the circuit (m=network.num_qubits)

    Args:
    num_qubits: int
    num_training_pairs (optional, default: None): int
    num_validation_pairs (optional, default: None): int
    network_unitary (Optional[Union[np.ndarray,List[complex],str]], optional): Network unitary. Defaults to None.
    input_training_states (Optional[Union[List[np.ndarray],str]], optional): Input states to generate training pairs. Defaults to None.
    input_validation_states (Optional[Union[List[np.ndarray],str]], optional): Input states to generate validation pairs. Defaults to None.


    Returns:
    training pairs: list[[QubitState, Unitary*QubitState]]
    validation pairs: list[[QubitState, Unitary*QubitState]]
    network_unitary: np.ndarray
    """
    if (num_training_pairs is None): num_training_pairs = 2**num_qubits
    if (num_validation_pairs is None): num_validation_pairs = 2**num_qubits
    if (network_unitary is None): network_unitary = randomQubitUnitary(num_qubits)
    if (isinstance(network_unitary, str)): 
        network_unitary = np.asarray(np.loadtxt(network_unitary, dtype=complex))
    assert len(network_unitary) == 2**num_qubits, "Dimension of network unitary {} does not match network architecture".format(np.shape(network_unitary))
    
    # load training states
    input_states: List[np.ndarray] = []
    if (input_training_states is None):
        input_states = [randomQubitState(num_qubits) for _ in range(num_training_pairs)]
    elif (isinstance(input_training_states, str)):
        input_states = [np.asarray(input_state) for input_state in np.loadtxt(input_training_states, dtype=complex)]
        sd.save_execution_info(load_training_states_from=input_training_states)
    else:
        input_states = input_training_states
    assert len(input_states) >= num_training_pairs, 'Number of input training states is smaller than given number of training pairs.'
    assert all((len(input_state) == 2**num_qubits) for input_state in input_states), 'Dimension of input training states does not match network architecture'

    # load validation states
    input_val_states: List[np.ndarray] = []
    if (input_validation_states is None):
        input_val_states = [randomQubitState(num_qubits) for _ in range(num_validation_pairs)]
    elif (isinstance(input_validation_states, str)):
        input_val_states = [np.asarray(input_state) for input_state in np.loadtxt(input_validation_states, dtype=complex)]
        sd.save_execution_info(load_validation_states_from=input_validation_states)
    else:
        input_val_states = input_validation_states
    assert len(input_val_states) >= num_validation_pairs, 'Number of input validation states is smaller than given number of validation pairs.'
    assert all((len(input_val_state) == 2**num_qubits) for input_val_state in input_val_states), 'Dimension of input validation states does not match network architecture'

    train_and_val_pairs = []
    for i in range(num_training_pairs):
        output_state = network_unitary.dot(input_states[i])
        train_and_val_pairs.append([input_states[i], output_state])
    for i in range(num_validation_pairs):
        output_state = network_unitary.dot(input_val_states[i])
        train_and_val_pairs.append([input_val_states[i], output_state])
    return np.array(train_and_val_pairs[:num_training_pairs]), np.array(train_and_val_pairs[num_training_pairs:]), network_unitary
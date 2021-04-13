# own modules
import save_data as sd # type: ignore
import training

# --- QISKIT ---
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, Aer, execute, assemble # type: ignore
from qiskit.quantum_info.operators import Operator # type: ignore
from qiskit.circuit import ParameterVector # type: ignore
from qiskit.aqua.operators.primitive_ops import MatrixOp # type: ignore
from qiskit.providers.ibmq.managed import IBMQJobManager # type: ignore
import qiskit.providers.aer.noise as noise # type: ignore

# additional math libs
import numpy as np # type: ignore
from scipy.constants import pi # type: ignore
from tenpy.linalg.random_matrix import GUE # type: ignore

from typing import Union, Optional, List, Tuple, Any, Dict
import itertools

import logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------

class Network_DQNN:
    """
    Architecture #9: https://www.notion.so/9-CAN-gates-applied-directly-to-the-corresponding-qubits-without-swaps-e1a16986754d4dfebc3bad33bac939a5
    """
    def __init__(self, 
                qnn_arch: List[int],
                fidelity_measurement_method: str = 'big_swap',
                params: Optional[str] = None):
        """
        Initialized the network class.

        Args:
            qnn_arch (List[int]): The QNN architecture (e.g. [2,2]).
            fidelity_measurement_method (str, optional): The fidelity measurement method. Defaults to 'big_swap'.
            params (Optional[str], optional): The initial network parameters. Defaults to None (will be randomly generated).
        """        
        assert qnn_arch[0]==qnn_arch[-1], "Not a valid QNN-Architecture."
        self.fid_meas_method = fidelity_measurement_method
        self.qnn_arch = qnn_arch
        self.num_qubits = qnn_arch[0]
        self.num_params = sum([qnn_arch[l]*qnn_arch[l+1]*3 + (qnn_arch[l])*3 for l in range(len(qnn_arch)-1)]) + qnn_arch[-1]*3 # total number of parameters
        self.params_per_layer = [self.qnn_arch[l]*self.qnn_arch[l+1]*3 + (self.qnn_arch[l])*3 for l in range(len(qnn_arch)-1)] # number of parameters per layer
        self.params = generate_network_parameters(num_params=self.num_params, load_from=params)
        self.auxillary_qubits = 0 if fidelity_measurement_method == "destructive_swap" else 1 # 0 for destructive_swap, 1 for big_swap
        self.required_qubits = sum(self.qnn_arch)+self.qnn_arch[0]+self.auxillary_qubits # required number of qubits
        # circuit types
        self.id_circuits: List[Any] = [] # identity circuits
        self.fid_circuits: List[Any] = [] # fidelity circuits
        self.tp_circuits: List[Any] = [] # training pair circuits
        self.vp_circuits: List[Any] = [] # validation pair circuits
        self.param_vector: List[Any] = [] # parameter vector (network parameters)
        # gate nosie study
        self.gate_error_probabilities: dict = {}
        self.noise_model: Any = None
        self.coupling_map: Any = None
        
    def __str__(self):
        """
        The network's description.

        Returns:
            string: Network description.
        """        
        return "Network_DQNN of the form: {}".format(self.qnn_arch)
        
    def update_params(self, params: Optional[Union[str, List[float]]] = None):
        """
        Update the network parameters.

        Args:
            params (Optional[Union[str, List[float]]], optional): The new network parameters. Defaults to None.
        """        
        if params is not None:
            # if params are not given -> initialize them randomly
            self.params = generate_network_parameters(num_params=self.num_params, load_from=params) if (isinstance(params,str)) else params 
        else:
            logger.warning('Params could not be updated because given params are None.')
          
    def circuit(self,
                state_pair: List[np.ndarray],
                params: Union[List[float], Any],
                draw_circ: bool = False) -> QuantumCircuit:
        """
        Creates the quantum circuit.

        Args:
            state_pair (List[np.ndarray]): Training pairs (used for initialization).
            params (Union[List[float], Any]): Network parameters.
            draw_circ (bool, optional): Whether the circuit should be drawn. Defaults to False.

        Returns:
            QuantumCircuit: The quantum circuit.
        """    
        # initialize the quantum circuit
        circ, q_reg, c_reg = init_quantum_circuit(self.required_qubits, 2*self.num_qubits if self.fid_meas_method == "destructive_swap" else 1)
        # add initialization the input states (from state_pair)
        circ = add_input_state_initialization(circ, q_reg[self.auxillary_qubits:self.num_qubits*2+self.auxillary_qubits], self.num_qubits, state_pair)
        # going through each output layer
        for layer in range(len(self.qnn_arch)-1):
            # the resepctive parameters
            layer_params = params[np.sign(layer)*sum(self.params_per_layer[:layer]):sum(self.params_per_layer[:layer+1])]
            # the respective qubit register
            in_and_output_register = q_reg[self.num_qubits + self.auxillary_qubits + np.sign(layer)*sum(self.qnn_arch[:layer]):self.num_qubits + self.auxillary_qubits + sum(self.qnn_arch[0:layer+2])]
            # append subcircuit connecting all neurons of (layer+1) to layer
            circ.append(self.generate_canonical_circuit_all_neurons(layer_params, layer=layer+1, draw_circ=draw_circ).to_instruction(), in_and_output_register)
        # add last U3s to all output qubits (last layer)
        circ = add_one_qubit_gates(circ, q_reg[-self.qnn_arch[-1]:], params[-self.qnn_arch[-1]*3:])
        # add fidelity measurement
        circ = add_fidelity_measurement(circ, q_reg, c_reg, self.num_qubits, method=self.fid_meas_method)
        return circ
    
    def transpile_circuits(self,
                           circuits: List[QuantumCircuit],
                           backend: Any,
                           optimization_level: int = 3,
                           draw_circ: bool = False) -> List[QuantumCircuit]:
        """
        Transpiles the given circuits.

        Args:
            circuits (List[QuantumCircuit]): The QuantumCircuits which should be transpiled.
            backend (Any): The Backend for which the transpilation should be optimized.
            optimization_level (int, optional): The optimization level of the transpilation. Defaults to 3.
            draw_circ (bool, optional): Whether the transpiled circuit should be drawn. Defaults to False.

        Returns:
            List[QuantumCircuit]: The transpiled quantum circuits.
        """
        # set the backend, coupling map, basis gates and optimization level of gate_error_probabilities (if given)
        transpile_backend = backend if not self.gate_error_probabilities else None
        transpile_coupling_map = None if not self.gate_error_probabilities else self.coupling_map
        transpile_basis_gates = None if not self.gate_error_probabilities else self.noise_model.basis_gates
        # optimization level should be ever 0,1,2 or 3
        if not optimization_level in [0,1,2,3]: 
            logger.warning("Optimization level out of bounds. An optimization level of 3 will be used.")
            optimization_level = 3
        transpile_optlvl = 1 if backend.name() == 'qasm_simulator' and not self.gate_error_probabilities else optimization_level
        # transpile the quantum circuits
        transpiled_circuits = transpile(circuits, backend=transpile_backend, 
            optimization_level=transpile_optlvl, coupling_map=transpile_coupling_map, 
            basis_gates=transpile_basis_gates, seed_transpiler=0)
        # function should return a list of quantum circuits
        if not isinstance(transpiled_circuits, list): transpiled_circuits = [transpiled_circuits]
        if draw_circ:
            # draw a single circuit 
            sd.draw_circuit(circuits[0], filename='circuit.png')
            # draw a single transpiled circuits
            sd.draw_circuit(transpiled_circuits[0], filename='transpiled_circuit.png')
            # save the depth and number of operations of the transpiled circuit
            sd.save_execution_info(transpilation_info="depth: {}, count_ops: {}".format(transpiled_circuits[0].depth(), transpiled_circuits[0].count_ops()))
        return transpiled_circuits
    
    def execute_circuits(self,
                         circuits: List[QuantumCircuit], 
                         backend: Any, 
                         shots: int = 2**14, 
                         **unused_args: Any) -> Any:
        """
        Executes the QuantumCircuits using the Backend.

        Args:
            circuits (List[QuantumCircuit]): The QuantumCircuits which should be executed.
            backend (Any): The backend for the execution.
            shots (int, optional): The number of shots used for the execution. Defaults to 2**14.

        Returns:
            Any: The measurement results (list of counts).
        """        
        if not self.gate_error_probabilities:
            if "qasm_simulator" in backend.name():
                # simulator or simulated backend
                qobj_circuits = assemble(circuits, backend=backend, shots=shots)
                job = backend.run(qobj_circuits)
                return job.result().get_counts()
            # real device execution
            # use the IBMQJobManager to break the circuits into multiple jobs
            job_manager = IBMQJobManager()
            job = job_manager.run(circuits, backend=backend, shots=shots)
            result = job.results()
            return [result.get_counts(i) for i in range(len(circuits))]
        # gate_error_probability is given -> use the specific function
        return execute_noise_simulation(self, circuits, self.gate_error_probabilities, shots)

    def generate_canonical_circuit_all_neurons(self,
                                               params: List[Any],
                                               layer: int,
                                               draw_circ: bool = False) -> QuantumCircuit:  
        """
        Creates a QuantumCircuit containing the parameterized CAN gates (plus single qubit U3 gates).
        The definition of the CAN gates is taken from https://arxiv.org/abs/1905.13311.
        The parameters should have length self.qnn_arch[0]*self.qnn_arch[1]*6 + self.qnn_arch[1]*6.

        Args:
            params (List[Any]): List of parameters for the parametrized gates. ParameterVector or List[float].
            layer (int): Index of the current output layer.
            draw_circ (bool, optional): Whether the sub-circuit should be drawn. Defaults to False.

        Returns:
            QuantumCircuit: Quantum circuit containing all the parameterized gates of the respective layer.
        """
        # sub-architecture of the layer (length 2)
        qnn_arch = self.qnn_arch[layer-1:layer+1]
        # number of qubits required for the layer
        num_qubits = qnn_arch[0]+qnn_arch[1]
        
        # initialize the quantum circuit
        circ, q_reg, _ = init_quantum_circuit(num_qubits, name="Layer {}".format(layer))
        
        # add U3s to input qubits
        circ = add_one_qubit_gates(circ, q_reg[:qnn_arch[0]], params)
        
        # loop over all neurons
        for i in range(qnn_arch[1]):
            # parameters of the respective "neuron gates"
            # (can be larer than needed, overflow will be ignored)
            neuron_params = params[qnn_arch[0]*3 + qnn_arch[0]*3*i:]
            # iterate over all input neurons and apply CAN gates
            for j in range(qnn_arch[0]):
                tx, ty, tz = neuron_params[j*3:(j+1)*3]
                circ.rxx(2*tx, q_reg[j], q_reg[qnn_arch[0]+i])
                circ.ryy(2*ty, q_reg[j], q_reg[qnn_arch[0]+i])
                circ.rzz(2*tz, q_reg[j], q_reg[qnn_arch[0]+i])

        if (draw_circ):
            # draw the sub-circuit
            sd.draw_circuit(circ, filename='circuit-layer_{}.png'.format(layer))
        return circ

    def identity_network_parameters(self, param_value: float = 0.0) -> List[float]:
        """
        Returns the identity network parameters, i.e. parameters such that the network 
        acts as the identity (only swaps the input state to the output qubits).
        The param_value can be set to 0.01 (for example) in order to avoid gate reduction in the transpilation.
        WARNING: Only works for networks with constant layer-sizes, e.g. [2,2,2].

        Args:
            param_value (float, optional): Numerical value for the gates which should act as the identity. Defaults to 0.0.

        Returns:
            List[float]: A list of parameter such that the network acts as the identity.
        """        
        params: List[float] = []
        # loop over each layer
        for l in range(len(self.qnn_arch)-1):
            # U3(0,0,0) = identity
            params += [param_value]*self.qnn_arch[l]*3
            # loop over each output neuron
            for j1 in range(self.qnn_arch[l+1]):
                # loop over each input neuron
                for j2 in range(self.qnn_arch[l]):
                    if (j1==j2):
                        # perform swap -> SWAP = CAN(pi/4, pi/4, pi/4)
                        params += [pi/4]*3
                    else:
                        # CAN(0,0,0) = identity
                        params += [param_value]*3
        # last U3s
        params += [param_value]*3*self.qnn_arch[-1]
        return params
    
    def set_identity_params(self):
        """
        Sets the network parameters to identity network parameters.
        """        
        self.params = self.identity_network_parameters()
        
    def reshape_params_for_plotting(self,
                                    param_per_epoch: List[List[float]]) -> List[List[dict]]:
        """
        Reshapes the list of parameter-lists. Includes information about the plotting title and values for parameter plotting.
        The return value is used for the specific plotting routine.

        Args:
            param_per_epoch (Any): List of parameters per training epoch.

        Returns:
            List[List[Dict]]: [[{title: "Layer 1, Preparation", params: [params per epoch]}, ... (neuronwise)], ... (layerwise)]
        """
        all_layers: List[List[dict]] = []
        # go through each output layer
        for layer in range(len(self.qnn_arch)-1):
            all_neurons: List[dict] = []
            # parameters of this layer (per training epoch)
            layer_params = param_per_epoch[np.sign(layer)*sum(self.params_per_layer[:layer]):sum(self.params_per_layer[:layer+1])]
            for j in range(self.qnn_arch[layer+1]+1):
                neuron_dict: dict = {}
                if j == 0:
                    # first layer (includes U3 gates)
                    neuron_dict['title'] = 'Layer {}, Preparation'.format(layer+1)
                    neuron_dict['params'] = layer_params[:self.qnn_arch[layer]*3]
                else:
                    neuron_dict["title"] = 'Layer {}, Neuron {}'.format(layer+1, j)
                    neuron_dict['params'] = layer_params[self.qnn_arch[layer]*3:][3*self.qnn_arch[layer]*(j-1):3*self.qnn_arch[layer]*j]
                    if layer == len(self.qnn_arch)-2:
                        # last layer (includes last U3 gates)
                        if j-1 < self.qnn_arch[-1]-1:
                            neuron_dict['params'] += param_per_epoch[-(self.qnn_arch[-1]-(j-1))*3:-(self.qnn_arch[-1]-(j-1)-1)*3]
                        else:
                            neuron_dict['params'] += param_per_epoch[-3:]
                all_neurons.append(neuron_dict)
            all_layers.append(all_neurons)      
        return all_layers          

# -----------------------------------------------------------------------------

class Network_QAOA:
    """
    Architecture $1: https://www.notion.so/1-QAOA-network-1fc9d7417df6404a8a6b4570dc5c7ae9
    """
    def __init__(self, 
                num_qubits: int,
                fidelity_measurement_method: str = 'big_swap',
                num_params_fac: Union[int,float] = 1,
                params: Optional[str] = None,
                A: Optional[Any] = None,
                B: Optional[Any] = None):
        """
        The Quantum Approximate Optimization Algorithm.

        Args:
            num_qubits (int): The number of required qubits.
            fidelity_measurement_method (str, optional): The fidelity measyrement method. Defaults to 'big_swap'.
            num_params_fac (Union[int,float], optional): Scales the total number of parameters (2*2^(2*num_qubtis*num_params_fac/2). Defaults to 1.
            params (Optional[str], optional): The network parameters. Defaults to None.
            A (Optional[Any], optional): The A-matrix of the QAOA. Defaults to None.
            B (Optional[Any], optional): The B-matrix of the QAOA. Defaults to None.
        """        
        self.fid_meas_method = fidelity_measurement_method
        self.num_qubits = num_qubits
        self.num_params = int(2**(2*num_qubits) * num_params_fac/2)*2 # round to the next even number smaller or equal
        self.qnn_arch = [num_qubits]*(self.num_params//2+1)
        self.params_per_layer = [2]*(len(self.qnn_arch)-1)
        self.params = generate_network_parameters([-1,1], self.num_params, load_from=params)
        self.auxillary_qubits = 0 if fidelity_measurement_method == "destructive_swap" else 1
        self.required_qubits = 2*num_qubits + self.auxillary_qubits # number of required qubits
        self.update_matrices_AB(A, B) # set self.A and self.B if given, else generate from GUE
        # circuit types
        self.id_circuits: List[Any] = [] # identity circuits
        self.fid_circuits: List[Any] = [] # fidelity circuits
        self.tp_circuits: List[Any] = [] # training pair circuits
        self.vp_circuits: List[Any] = [] # validation pair circuits
        self.param_vector: List[Any] = [] # List of network parameters (or ParameterVector)
        self.initial_layout: Any = None # layout of circuit transpilation
        # gate noise study
        self.gate_error_probabilities: dict = {}
        self.noise_model: Any = None
        self.coupling_map: Any = None
        
    def __str__(self):
        """
        The network's description.

        Returns:
            string: Network description.
        """        
        return "Network_QAOA with {} qubits and {} layers".format(self.num_qubits, len(self.qnn_arch)-1)
        
    def update_params(self, params: Optional[Union[str, List[float]]] = None):
        """
        Update the network parameters.

        Args:
            params (Optional[Union[str, List[float]]], optional): The new network parameters. Defaults to None.
        """        
        if params is not None:
            # if parameters are not given -> initialize them randomly
            self.params = generate_network_parameters(num_params=self.num_params, load_from=params) if (isinstance(params,str)) else params 
        else:
            logger.warning('Params could not be updated because given params are None.')
            
    def update_matrices_AB(self, A: Any = None, B: Any = None):
        """
        Sets the A and B matrix of the QAOA if given. If not given, it generates them from the GUE.
        The QAOA features a sequence of alternating operators e-itA and e-itauB.

        Args:
            A (Any, optional): A matrix of the QAOA (string to the file or matrix). Defaults to None.
            B (Any, optional): B matrix of the QAOA (string to the file or matrix). Defaults to None.
        """        
        
        # judgement free zone
        try:
            assert A != None 
            # A = qutip object of (get matrix from file if A is a string else set it to A)
            self.A = np.asarray(np.loadtxt(A, dtype=complex) if isinstance(A, str) else A)
        except:
            # generate A from the GUE
            self.A = generate_matrix_from_GUE(self.num_qubits)
        try:
            assert B != None
            # B = qutip object of (get matrix from file if B is a string else set it to B)
            self.B = np.asarray(np.loadtxt(B, dtype=complex) if isinstance(B, str) else B)
        except:
            # generate B from the GUE
            self.B = generate_matrix_from_GUE(self.num_qubits)

    def circuit(self,
                state_pair: List[np.ndarray],
                params: Union[List[float], Any],
                draw_circ: bool = False) -> QuantumCircuit:
        """
        Creates the quantum circuits.

        Args:
            state_pair (List[np.ndarray]): Training pairs (used for the initialization).
            params (Union[List[float], Any]): Network parameters.
            draw_circ (bool, optional): Whether the circuit should be drawn. Defaults to False.

        Returns:
            QuantumCircuit: The quantum circuit.
        """        
        # initialize the quantum circuit
        circ, q_reg, c_reg = init_quantum_circuit(self.required_qubits, 2*self.num_qubits if self.fid_meas_method == "destructive_swap" else 1)
        # add initialization of the input states (from training pair)
        circ = add_input_state_initialization(circ, q_reg[self.auxillary_qubits:], self.num_qubits, state_pair)
        # quantum register of the network part
        q_register = q_reg[self.num_qubits + self.auxillary_qubits:]
        # going through each output layer
        for layer in range(len(self.qnn_arch)-1):
            # the respective parameters
            layer_params = params[np.sign(layer)*sum(self.params_per_layer[:layer]):sum(self.params_per_layer[:layer+1])]
            # apply e-itA
            UA = ((MatrixOp(self.A) * layer_params[0]).exp_i()).to_circuit()
            UA.name = r'$e^{-i t_{%s} A}$' % str(layer+1)
            # apply e-itauB
            UB = ((MatrixOp(self.B) * layer_params[1]).exp_i()).to_circuit()
            UB.name = r'$e^{-i \tau _{%s} B}$' % str(layer+1)
            # append to circuit
            circ.append(UA, q_register)
            circ.append(UB, q_register)
        # add fidelity measurement
        circ = add_fidelity_measurement(circ, q_reg, c_reg, self.num_qubits, method=self.fid_meas_method)
        return circ
    
    def transpile_circuits(self,
                           circuits: List[QuantumCircuit],
                           backend: Any,
                           optimization_level: int = 3,
                           draw_circ: bool = False) -> List[QuantumCircuit]:
        """
        Transpiles the given circuits.

        Args:
            circuits (List[QuantumCircuit]): The QuantumCircuits which should be transpiled.
            backend (Any): The Backend for which the transpilation should be optimized.
            optimization_level (int, optional): The optimization level of the transpilation. Defaults to 3.
            draw_circ (bool, optional): Whether the transpiled circuit should be drawn. Defaults to False.

        Returns:
            List[QuantumCircuit]: The transpiled quantum circuits.
        """        
        if draw_circ: 
            # draw a single circuit
            sd.draw_circuit(circuits[0], filename='circuit.png')
        # set the backend, coupling map, basis gates and optimization level of gate_error_probabilities (if given)
        transpile_backend = backend if not self.gate_error_probabilities else None
        transpile_coupling_map = None if not self.gate_error_probabilities else self.coupling_map
        transpile_basis_gates = None if not self.gate_error_probabilities else self.noise_model.basis_gates
        transpile_optlvl = 1 if backend.name() == 'qasm_simulator' and not self.gate_error_probabilities else optimization_level
        if self.initial_layout is None and backend.name() != "qasm_simulator":
            # initial layout not set yet
            logger.info('Transpile circuit with optimization level 3 to generate the initial layout.')
            # transpile the circuit to get the initial layout
            trial_circuit = transpile(circuits[0].assign_parameters({self.param_vector: self.params}), 
                backend=transpile_backend, optimization_level=transpile_optlvl, 
                coupling_map=transpile_coupling_map, basis_gates=transpile_basis_gates, seed_transpiler=0)
            self.initial_layout = trial_circuit._layout
            if draw_circ:
                # draw the transpiled circuit
                sd.draw_circuit(trial_circuit, filename='transpiled_circuit_optlvl3.png')
                transpiled_circuit_for_drawing = transpile(circuits[0].assign_parameters({self.param_vector: self.params}), 
                    backend=transpile_backend, optimization_level=1, coupling_map=transpile_coupling_map, basis_gates=transpile_basis_gates, 
                    initial_layout=self.initial_layout, seed_transpiler=0)
                sd.draw_circuit(transpiled_circuit_for_drawing, filename='transpiled_circuit_optlvl1.png')
                # save the depth and number of operations of the transpiled circuit
                sd.save_execution_info(transpilation_info="depth: {}, count_ops: {}".format(transpiled_circuit_for_drawing.depth(), transpiled_circuit_for_drawing.count_ops()))
        return circuits
    
    def execute_circuits(self,
                         circuits: List[QuantumCircuit], 
                         backend: Any, 
                         shots: int = 2**14, 
                         **unused_args: Any) -> Any:
        """
        Executes the QuantumCircuit using the Backend.

        Args:
            circuits (List[QuantumCircuit]): The QuantumCircuits which should be executed..2e
            backend (Any): The backend for the execution.
            shots (int, optional): The number of shots used for the execution. Defaults to 2**14.

        Returns:
            Any: The measurement results (list of counts).
        """        
        if not self.gate_error_probabilities:
            if "qasm_simulator" in backend.name():
                # simulator or simulated device
                # only for simulated device (opt_level=0) or pure simulator (opt_level=1)
                job = execute(circuits, backend=backend, shots=shots, optimization_level=int(backend.name() == "qasm_simulator"), initial_layout=self.initial_layout, seed_transpiler=0)
                return job.result().get_counts()
            # real device execution
            # transpile using optimization level 1 to not reduce the QAOA gates
            # (optimization level 3 is doing pre-calculations of the QAOA sequence)
            circuits = transpile(circuits, backend=backend, optimization_level=1, initial_layout=self.initial_layout, seed_transpiler=0)
            # use the IBMQJobManager to break the param_circs into multiple jobs
            job_manager = IBMQJobManager()
            job = job_manager.run(circuits, backend=backend, shots=shots)
            result = job.results()
            return [result.get_counts(i) for i in range(len(circuits))]
            
        # gate_error_probability is given -> use the specific function
        return execute_noise_simulation(self, circuits, self.gate_error_probabilities, shots)
    

    def identity_network_parameters(self, param_value: float = 0.0001) -> List[float]:
        """
        Returns the identity network parameters, i.e. patameters such that the networks
        acts as the identity. The param_value is set to 0.0001 per default in order to
        avoid gate reductions by qiskit's transpile (the transpiler sees identity gates and thus
        ignores them, that is not what we want since we want to include the gate noise into the 
        measurement results).

        Args:
            param_value (float, optional): Numerical value for the gates which should act as the identity. Defaults to 0.0001.

        Returns:
            List[float]: A list of parameters such that the network acts as the identity.
        """        
        return [param_value]*self.num_params
    
    def set_identity_params(self):
        """
        Sets the network parameters to identity network parameters.
        """        
        self.params = self.identity_network_parameters()
    
    def reshape_params_for_plotting(self,
                                    param_per_epoch: List[List[float]]) -> List[List[dict]]:
        """
        Reshapes the list of parameter-lists. Includes information about the plotting title and values for parameter plotting.
        The return value is used for the specific plotting routine.

        Args:
            param_per_epoch (Any): List of parameters per training epoch.

        Returns:
            List[List[Dict]]: [[{title: "Layer 1, Preparation", params: [params per epoch]}, ... (neuronwise)], ... (layerwise)]
        """
        # number of figures
        num_boxes_side = 2**self.num_qubits//2
        # number of parameters per figure
        num_params_per_box = self.num_params//(num_boxes_side**2)
        all_layers: List[List[dict]] = []
        # iterate through the figures
        for i in range(num_boxes_side):
            all_neurons: List[dict] = []
            for j in range(num_boxes_side):
                # dict containing the figure title and data
                neuron_dict: dict = {
                    "title": "Layer {} & {}".format(num_boxes_side**2*i + j*num_boxes_side+1, num_boxes_side**2*i + j*num_boxes_side+2),
                    "params": param_per_epoch[(num_boxes_side*i + j) * num_params_per_box:(num_boxes_side*i + j+1) * num_params_per_box]
                }
                all_neurons.append(neuron_dict)
            all_layers.append(all_neurons)
        return all_layers

# -----------------------------------------------------------------------------

def generate_matrix_from_GUE(num_qubits: int) -> np.ndarray:
    """
    Randomly generates a matrix from the GUE.

    Args:
        num_qubits (int): Number of qubits. Dimention of the returned matrix is 2^num_qubits.

    Returns:
        np.ndarray: The matrix generated from the GUE.
    """    
    d = 2**num_qubits # dimension
    matrix = GUE((d,d))
    return matrix

def generate_network_parameters(param_range: Optional[Union[float, List[float]]] = 2*pi,
                                num_params: Optional[int] = None,
                                load_from: Optional[str] = None) -> List[float]:
    """
    Generate random network parameters (in specific range and with specific length). They are used for the parameterized gates.

    Args:
        param_range (Optional[Union[float, List[float]]], optional): Range of the random parameters. Defaults to 2*pi.
        num_params (Optional[int], optional): Number of parameters. Defaults to None.
        load_from (Optional[str], optional): Name of a file containing the parameters. Defaults to None.

    Returns:
        List[float]: The generated network parameters.
    """    
    if load_from:
        # filename is given -> load parameter from file
        all_params = np.loadtxt(load_from)
        if len(np.shape(all_params)) < 2: all_params = np.array([all_params])
        # length of loaded parameters, should match num_params
        if len(all_params[0][1:]) != num_params:
            logger.error("Loaded parameters have different size ({}) than expected ({})."
                .format(len(all_params[0][1:]), num_params))
            raise
        # if more than one paremeter list is given: display warning
        if (len(all_params) > 0): logger.warning('Your loaded parameters {} have more than one parameter set available. Make sure you really want to load the parameters from epoch 0.'.format(load_from))
        return all_params[0][1:]
    if isinstance(param_range, list):
        # param_range is a list -> consists of lower and upper bound
        return np.random.uniform(low=param_range[0], high=param_range[1], size=(num_params)) # Range of parameters, e.g. [-pi,pi] or [-1,1]
    # param_range is a float -> range = [0, param_range]
    return np.random.uniform(high=param_range, size=(num_params))


def construct_noise_model(network: Union[Network_DQNN, Network_QAOA]) -> None:
    """
    Constructs the noise model for the gate_error_probabilities of the respectice network.

    Args:
        network (Union[Network_DQNN, Network_QAOA]): The network's class. 
    """    
    provider = training.get_provider()
    backend = provider.get_backend('ibmq_16_melbourne')
    network.coupling_map = backend.configuration().coupling_map
        
    noise_model = noise.NoiseModel(["cx", "rz", "sx", "x"])
    for gate, value in network.gate_error_probabilities.items():
        error = noise.depolarizing_error(*value)
        noise_model.add_all_qubit_quantum_error(error, gate)
    network.noise_model = noise_model


def construct_and_transpile_circuits(network: Union[Network_DQNN, Network_QAOA],
                                     training_pairs: List[List[np.ndarray]],
                                     validation_pairs: List[List[np.ndarray]],
                                     optimization_level: int = 3,
                                     device_name: Optional[str] = None,
                                     simulator: Optional[bool] = None) -> None:
    """
    Constructs and transpiles the quantum circuits for all given training and validation pairs.

    Args:
        network (Union[Network_DQNN, Network_QAOA]): The network's class.
        training_pairs (List[List[np.ndarray]]): The list of training pairs.
        validation_pairs (List[List[np.ndarray]]): The list of validation pairs.
        optimization_level (int, optional): The optimization level used for the transpilation. Defaults to 3.
        device_name (Optional[str], optional): The name of the backend. Defaults to None.
        simulator (Optional[bool], optional): Whether a simulator (or simulated device) should be used. Defaults to None.
    """    
    if simulator:
        # simulator or simulated device
        device = training.DEVICE or training.get_device(network, bool(simulator), device_name=device_name, do_calibration=False)
        training.DEVICE = device
    else:
        # real device
        device = training.get_device(network, bool(simulator), device_name=device_name, do_calibration=False)
    
    # parameter initialization
    network.param_vector = ParameterVector("p", network.num_params)
    
    # network circuits for training pairs
    tp_circs = [network.circuit(tp, network.param_vector, draw_circ=(i==0)) for i,tp in enumerate(training_pairs)]
    network.tp_circuits = network.transpile_circuits(tp_circs, backend=device, optimization_level=optimization_level, draw_circ=True)
    
    # network circuits for validation pairs
    vp_circs = [network.circuit(vp, network.param_vector) for vp in validation_pairs]
    network.vp_circuits = network.transpile_circuits(vp_circs, backend=device, optimization_level=optimization_level, draw_circ=False)
    
    # identity circuit for identity cost
    id_circs = [network.circuit([tp[1], tp[1]], network.param_vector) for tp in training_pairs]
    network.id_circuits = network.transpile_circuits(id_circs, backend=device, optimization_level=optimization_level, draw_circ=False)

    # fidelity measurement circuit (unparameterized)
    fid_circs = [fidelity_circuit(network, [tp[1], tp[1]]) for tp in training_pairs]
    network.fid_circuits = network.transpile_circuits(fid_circs, backend=device, optimization_level=optimization_level, draw_circ=False)


def execute_noise_simulation(network: Union[Network_DQNN, Network_QAOA],
                             circuits: List[Any],
                             gate_error_probabilities: dict, 
                             shots: int) -> Any:
    """
    Execute the quantum circuits under consideration of the noise model.

    Args:
        network (Union[Network_DQNN, Network_QAOA]): The network's class.
        circuits (List[Any]): The quantum circuits which should be executed.
        gate_error_probabilities (dict): The gate error probabilities.
        shots (int): The number of shots used for the execution.

    Returns:
        Any: The measurement result (list of counts).
    """    
    # if network is QAOA -> use computed layout
    initial_layout = network.initial_layout if isinstance(network, Network_QAOA) else None
    # execute the quantum circuits
    job = execute(circuits, Aer.get_backend('qasm_simulator'), basis_gates=network.noise_model.basis_gates, initial_layout=initial_layout,
        noise_model=network.noise_model, shots=shots, coupling_map=network.coupling_map, seed_transpiler=0)
    return job.result().get_counts()


def init_quantum_circuit(num_qubits, num_cbits=0, name: Optional[str] = None) -> Tuple[QuantumCircuit, QuantumRegister, ClassicalRegister]:
    """
    Initializes a QuantumCircuit using num_qubits qubits and num_cbits classical bits.

    Args:
        num_qubits ([type]): The number of qubits.
        num_cbits (int, optional): The number of classical bits. Defaults to 0.
        name (Optional[str], optional): The quantum circuit's name. Defaults to None.

    Returns:
        Tuple[QuantumCircuit, QuantumRegister, ClassicalRegister]: The initialized QuantumCircuit and its QuantumRegister and ClassicalRegister
    """    
    # init register for quantum and classical bits
    q_register = QuantumRegister(num_qubits, 'q')
    c_register = ClassicalRegister(num_cbits, 'c') if num_cbits > 0 else None

    # init quantum circuit
    circ = QuantumCircuit(num_qubits, num_cbits, name=name) if c_register else QuantumCircuit(num_qubits, name=name)
    return circ, q_register, c_register

def add_input_state_initialization(circ: QuantumCircuit,
                                   q_reg: QuantumRegister,
                                   num_qubits_per_state: int,
                                   state_pair: List[np.ndarray]) -> QuantumCircuit:
    """
    Adds the state initialization to the given QuantumCircuit.

    Args:
        circ (QuantumCircuit): The quantum circuit.
        q_reg (QuantumRegister): The quantum register (qubits for the initialization).
        num_qubits_per_state (int): The number of qubits for each state.
        state_pair (List[np.ndarray]): A pair of states.

    Returns:
        QuantumCircuit: The given quantum circuit including state initializations.
    """        
    # initialize qubits
    circ.initialize(state_pair[1],[q_reg[i] for i in range(num_qubits_per_state)])
    circ.initialize(state_pair[0],[q_reg[num_qubits_per_state+i] for i in range(num_qubits_per_state)])
    return circ

def add_one_qubit_gates(circ: QuantumCircuit, 
                        q_reg: QuantumRegister, 
                        params: List[float],
                        u3: bool = True) -> QuantumCircuit:
    """
    Adds U3 gates (if u3=True else RX, RY, RX gates) to each qubit in the quantum register.

    Args:
        circ (QuantumCircuit): The quantum circuit.
        q_reg (QuantumRegister): The quantum register containing the qubits.
        params (List[flaot]): List of parameters (used for the one qubit gates). Should be a multiple of 3.
        u3 (bool): Whether U3 gates should be used. Defaults to True.

    Returns:
        QuantumCircuit: The given quantum circuit including the application of one qubit gates.
    """
    for i, qubit in enumerate(q_reg):
        if u3:
            circ.u(params[i*3], params[i*3+1], params[i*3+2], qubit)
        else:
            circ.rx(params[i*3], qubit)
            circ.ry(params[i*3+1], qubit)
            circ.rx(params[i*3+2], qubit)
    return circ

def add_fidelity_measurement(circ: QuantumCircuit,
                             q_register: QuantumRegister,
                             c_register: ClassicalRegister,
                             num_qubits: int,
                             method: str = 'big_swap') -> QuantumCircuit:
    """
    Adds a fidelity measurement to the given QuantumCircuit.
    If method == "big_swap":
        Adds the BIGSWAP operation to the quantum circuit. The BIGSWAP is a 
        controlled SWAP operation where all the qubits of the first layer will
        be swapped with all the qubits of the second layer if the qubit in register
        0 (control qubit) is in state |1>.
    If method == "destructive_swap":
        Adds bell state measurements to the circuit. The fidelity can be calcualted
        using classical post-processing (see execute_circuits and get_cost_from_counts
        in training.py)

    Args:
        circ (QuantumCircuit): The quantum circuit.
        q_register (QuantumRegister): The quantum register containing the qubits.
        c_register (ClassicalRegister): The classical register (the measurement result is stored here).
        num_qubits (int): The number of qubits of the states.
        method (str, optional): The fidelity measurement method. Defaults to 'big_swap'.
    Returns:
        QuantumCircuit: The given quantum circuit including the fidelity measurement.
    """
    
    if method == "destructive_swap":
        # destructive swap
        for i in range(num_qubits):
            circ.cx(q_register[-1-i], q_register[num_qubits-i-1])
        for i in range(num_qubits):
            circ.h(q_register[-1-i])
        # measurement of input qubits
        for i in range(num_qubits):
            circ.measure(q_register[i], c_register[i])
        # measurement of output qubits
        for i in range(num_qubits):
            circ.measure(q_register[-num_qubits + i], c_register[-num_qubits + i])
    else:
        # big swap
        circ.h(q_register[0])
        # constrolled swaps between all input and output qubits
        for i in range(1, num_qubits+1):
            circ.cswap(q_register[0], q_register[i], q_register[-(num_qubits+1)+i])
        circ.h(q_register[0])
        # measurement of the ancillary qubit
        circ.measure(q_register[0], c_register[0])

    return circ

def fidelity_circuit(network: Union[Network_DQNN, Network_QAOA],
                     state_pair: List[np.ndarray],
                     draw_circ: bool = False) -> QuantumCircuit:
    """
    Creates the fidelity circuit. A QuantumCircuit which features only the state initialization
    and fidelity measurement. It is used to classify the noise of the device.

    Args:
        network (Union[Network_DQNN, Network_QAOA]): The network's class.
        state_pair (List[np.ndarray]): The state pairs.

    Returns:
        QuantumCircuit: The fidelity circuit.
    """    
    # initialization of the quantum circuit
    circ, q_reg, c_reg = init_quantum_circuit(2*network.num_qubits + network.auxillary_qubits, 2*network.num_qubits if network.fid_meas_method == "destructive_swap" else 1)
    # add state initialization
    circ = add_input_state_initialization(circ, q_reg[network.auxillary_qubits:], network.num_qubits, state_pair)
    # add fidelity measurement
    circ = add_fidelity_measurement(circ, q_reg, c_reg, network.num_qubits, method=network.fid_meas_method)
    if draw_circ:
        # dar circuit
        sd.draw_circuit(circ, filename='fid_circuit')
    return circ
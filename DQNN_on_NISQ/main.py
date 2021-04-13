from user_config import api_token
from qiskit import IBMQ # type: ignore

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from execution import *

if __name__ == "__main__":
    
    folder = ""
    
    # Load IBMQ account
    try:
        if api_token != '':
            IBMQ.enable_account(api_token)
        else:
            IBMQ.load_account()
    except:
        logger.error('Set your api_token in user_config.py or use IBMQ.save_account(your_api_token) on your machine once.')
        
            
    #################### Choose the runs you are interested in below ####################
    
    #### 1. Single training run with either DQNN or QAOA ####
    # Used for figure 3 in "Training Quantum Neural Networks on NISQ devices"
    
    ## a) Choose an input folder (unitary, training states, validation states, init parameters)
    # folder = "input/dqnn_2-2"
    # folder = "input/qaoa_2qubits"
    
    ## b) Choose settings for your training run
    single_execution(
        simulator=True,
        network=Network_DQNN(qnn_arch=[1,1], fidelity_measurement_method='destructive_swap'),
        # network=Network_QAOA(num_qubits=2, fidelity_measurement_method='destructive_swap', 
        #                      A=folder+'/matrix_A.txt' if folder else None, B=folder+'/matrix_B.txt' if folder else None),
        shots=2**13,
        epochs=200,
        epsilon=0.25,
        eta=0.5,
        order_in_epsilon=2,
        gradient_method='gradient_descent',
        validation_step=1,
        load_unitary_from=folder+'/unitary.txt' if folder else None,
        load_params_from=folder+'/params.txt' if folder else None,
        load_training_states_from=folder+'/input_training_states.txt' if folder else None,
        load_validation_states_from=folder+'/input_validation_states.txt' if folder else None,
        device_name="ibmq_casablanca",
        optimization_level=3
    )
    
    ####    2. Train DQNN and QAOA multiple times with different number of training pairs ####
    ####       and for random unitaries, states and start parameters                      ####
    
    # Used for generalization analysis in figure 2a in "Training Quantum Neural Networks on NISQ devices"

    # train_for_different_start_parameters(num_runs=20, num_training_pairs=[1,2,3,4])

    ####    3. Train DQNN and QAOA multiple times with different gate noise ####
    ####       and for random unitaries, states and start parameters        ####
    
    # Used for gate noise analysis in figure 2b in "Training Quantum Neural Networks on NISQ devices"
    
    # train_for_different_error_probabilities(num_runs=5)
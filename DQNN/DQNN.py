######  Code for Feedforward Quantum Neural Networks

### Package-imports, universal definitions and remarks
import scipy as sc
import qutip as qt
import pandas as pd
import random
from time import time
import sys
import matplotlib.pyplot as plt
import pickle
import os.path
import numpy as np
import networkx as nx

# ket states
qubit0 = qt.basis(2, 0)
qubit1 = qt.basis(2, 1)
# density matrices
qubit0mat = qubit0 * qubit0.dag()
qubit1mat = qubit1 * qubit1.dag()


###  Helper functions for the QNN-Code
def partialTraceRem(obj, rem):
    # prepare keep list
    rem.sort(reverse=True)
    keep = list(range(len(obj.dims[0])))
    for x in rem:
        keep.pop(x)
    res = obj;
    # return partial trace:
    if len(keep) != len(obj.dims[0]):
        res = obj.ptrace(keep);
    return res;


def partialTraceKeep(obj, keep):
    # return partial trace:
    res = obj;
    if len(keep) != len(obj.dims[0]):
        res = obj.ptrace(keep);
    return res;


def swappedOp(obj, i, j):
    if i == j: return obj
    numberOfQubits = len(obj.dims[0])
    permute = list(range(numberOfQubits))
    permute[i], permute[j] = permute[j], permute[i]
    return obj.permute(permute)


def tensoredId(N):
    # Make Identity matrix
    res = qt.qeye(2 ** N)
    # Make dims list
    dims = [2 for i in range(N)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    # Return
    return res


def tensoredQubit0(N):
    # Make Qubit matrix
    res = qt.fock(2 ** N).proj()  # For some reason ran faster than fock_dm(2**N) in tests
    # Make dims list
    dims = [2 for i in range(N)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    # Return
    return res


def unitariesCopy(unitaries):
    newUnitaries = []
    for layer in unitaries:
        newLayer = []
        for unitary in layer:
            newLayer.append(unitary.copy())
        newUnitaries.append(newLayer)
    return newUnitaries


### Random generation of unitaries, training data and networks


def randomQubitUnitary(numQubits):
    dim = 2 ** numQubits
    # Make unitary matrix
    res = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    res = sc.linalg.orth(res)
    res = qt.Qobj(res)
    # Make dims list
    dims = [2 for i in range(numQubits)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    # Return
    return res


def randomQubitState(numQubits):
    dim = 2 ** numQubits
    # Make normalized state
    res = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    res = (1 / sc.linalg.norm(res)) * res
    res = qt.Qobj(res)
    # Make dims list
    dims1 = [2 for i in range(numQubits)]
    dims2 = [1 for i in range(numQubits)]
    dims = [dims1, dims2]
    res.dims = dims
    # Return
    return res


def randomTrainingData(unitary, N):
    numQubits = len(unitary.dims[0])
    trainingData = []
    # Create training data pairs
    for i in range(N):
        t = randomQubitState(numQubits)
        ut = unitary * t
        trainingData.append([t, ut])
    # Return
    return trainingData


def randomNetwork(qnnArch, numTrainingPairs):
    # assert qnnArch[0] == qnnArch[-1], "Not a valid QNN-Architecture."

    # Create the targeted network unitary and corresponding training data
    networkUnitary = randomQubitUnitary(qnnArch[-1])
    networkTrainingData = randomTrainingData(networkUnitary, numTrainingPairs)

    # Create the initial random perceptron unitaries for the network
    networkUnitaries = [[]]
    for l in range(1, len(qnnArch)):
        numInputQubits = qnnArch[l - 1]
        numOutputQubits = qnnArch[l]

        networkUnitaries.append([])
        for j in range(numOutputQubits):
            unitary = randomQubitUnitary(numInputQubits + 1)
            if numOutputQubits - 1 != 0:
                unitary = qt.tensor(randomQubitUnitary(numInputQubits + 1), tensoredId(numOutputQubits - 1))
                unitary = swappedOp(unitary, numInputQubits, numInputQubits + j)
            networkUnitaries[l].append(unitary)

    # Return
    return (qnnArch, networkUnitaries, networkTrainingData, networkUnitary)


### QNN-Code


def costFunction(trainingData, outputStates):
    costSum = 0
    if len(trainingData) == 0:  # new
        return 1
    for i in range(len(trainingData)):
        costSum += trainingData[i][1].dag() * outputStates[i] * trainingData[i][1]
    return costSum.tr() / len(trainingData)


def makeLayerChannel(qnnArch, unitaries, l, inputState):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Tensor input state
    state = qt.tensor(inputState, tensoredQubit0(numOutputQubits))

    # Calculate layer unitary
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni

    # Multiply and tensor out input state
    return partialTraceRem(layerUni * state * layerUni.dag(), list(range(numInputQubits)))


def makeAdjointLayerChannel(qnnArch, unitaries, l, outputState):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Prepare needed states
    inputId = tensoredId(numInputQubits)
    state1 = qt.tensor(inputId, tensoredQubit0(numOutputQubits))
    state2 = qt.tensor(inputId, outputState)

    # Calculate layer unitary
    layerUni = unitaries[l][0].copy()
    for i in range(1, numOutputQubits):
        layerUni = unitaries[l][i] * layerUni

    # Multiply and tensor out output state
    return partialTraceKeep(state1 * layerUni.dag() * state2 * layerUni, list(range(numInputQubits)))


def feedforward(qnnArch, unitaries, trainingData):
    storedStates = []
    for x in range(len(trainingData)):
        currentState = trainingData[x][0] * trainingData[x][0].dag()
        layerwiseList = [currentState]
        for l in range(1, len(qnnArch)):
            currentState = makeLayerChannel(qnnArch, unitaries, l, currentState)
            layerwiseList.append(currentState)
        storedStates.append(layerwiseList)
    return storedStates


def makeUpdateMatrix(qnnArch, unitaries, trainingData, storedStates, lda, ep, l, j):
    numInputQubits = qnnArch[l - 1]

    # Calculate the sum:
    summ = 0
    for x in range(len(trainingData)):
        # Calculate the commutator
        firstPart = updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x)
        secondPart = updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x)
        mat = qt.commutator(firstPart, secondPart)

        # Trace out the rest
        keep = list(range(numInputQubits))
        keep.append(numInputQubits + j)
        mat = partialTraceKeep(mat, keep)

        # Add to sum
        summ = summ + mat

    # Calculate the update matrix from the sum
    summ = (-ep * (2 ** numInputQubits) / (lda * len(trainingData))) * summ
    return summ.expm()


def updateMatrixFirstPart(qnnArch, unitaries, storedStates, l, j, x):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Tensor input state
    state = qt.tensor(storedStates[x][l - 1], tensoredQubit0(numOutputQubits))

    # Calculate needed product unitary
    productUni = unitaries[l][0]
    for i in range(1, j + 1):
        productUni = unitaries[l][i] * productUni

    # Multiply
    return productUni * state * productUni.dag()


def updateMatrixSecondPart(qnnArch, unitaries, trainingData, l, j, x):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Calculate sigma state
    state = trainingData[x][1] * trainingData[x][1].dag()
    for i in range(len(qnnArch) - 1, l, -1):
        state = makeAdjointLayerChannel(qnnArch, unitaries, i, state)
    # Tensor sigma state
    state = qt.tensor(tensoredId(numInputQubits), state)

    # Calculate needed product unitary
    productUni = tensoredId(numInputQubits + numOutputQubits)
    for i in range(j + 1, numOutputQubits):
        productUni = unitaries[l][i] * productUni

    # Multiply
    return productUni.dag() * state * productUni


def makeUpdateMatrixTensored(qnnArch, unitaries, lda, ep, trainingData, storedStates, l, j):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    res = makeUpdateMatrix(qnnArch, unitaries, lda, ep, trainingData, storedStates, l, j)
    if numOutputQubits - 1 != 0:
        res = qt.tensor(res, tensoredId(numOutputQubits - 1))
    return swappedOp(res, numInputQubits, numInputQubits + j)


def qnnTraining(qnnArch, initialUnitaries, trainingData, lda, ep, trainingRounds, alert=0):
    ### FEEDFORWARD
    # Feedforward for given unitaries
    s = 0
    currentUnitaries = initialUnitaries
    storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

    # Cost calculation for given unitaries
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])
    plotlist = [[s], [costFunction(trainingData, outputStates)]]

    # Optional
    runtime = time()

    # Training of the Quantum Neural Network
    for k in range(trainingRounds):
        if alert > 0 and k % alert == 0: print("In training round " + str(k))

        ### UPDATING
        newUnitaries = unitariesCopy(currentUnitaries)

        # Loop over layers:
        for l in range(1, len(qnnArch)):
            numInputQubits = qnnArch[l - 1]
            numOutputQubits = qnnArch[l]

            # Loop over perceptrons
            for j in range(numOutputQubits):
                newUnitaries[l][j] = (
                        makeUpdateMatrixTensored(qnnArch, currentUnitaries, trainingData, storedStates, lda, ep, l,
                                                 j) * currentUnitaries[l][j])

        ### FEEDFORWARD
        # Feedforward for given unitaries
        s = s + ep
        currentUnitaries = newUnitaries
        storedStates = feedforward(qnnArch, currentUnitaries, trainingData)

        # Cost calculation for given unitaries
        outputStates = []
        for m in range(len(storedStates)):
            outputStates.append(storedStates[m][-1])
        plotlist[0].append(s)
        plotlist[1].append(costFunction(trainingData, outputStates))

    # Optional
    runtime = time() - runtime
    print("Trained " + str(trainingRounds) + " rounds for a " + str(qnnArch) + " network and " + str(
        len(trainingData)) + " training pairs in " + str(round(runtime, 2)) + " seconds")

    # Return
    return [plotlist, currentUnitaries]


### Semisupervised training


# semisupervised QNN training, outputs lists of the loss functions and the network unitaries
def qnnTrainingSsv(qnnArch, initialUnitaries, trainingData, trainingDataSv, listSv, trainingDataUsv, listUsv, lda, ep,
                   trainingRounds, alert=0):
    ### FEEDFORWARD
    # Feedforward for given unitaries
    s = 0
    N = len(trainingData)
    S = len(trainingDataSv)
    currentUnitaries = initialUnitaries
    storedStates = feedforward(qnnArch, currentUnitaries, trainingData)
    storedStatesSv = []
    for index in listSv:
        storedStatesSv.append(storedStates[index])
    storedStatesUsv = []
    for index in listUsv:
        storedStatesUsv.append(storedStates[index])

    # Cost calculation for given unitaries
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])

    outputStatesSv = []
    for k in range(len(storedStatesSv)):
        outputStatesSv.append(storedStatesSv[k][-1])

    outputStatesUsv = []
    for k in range(len(storedStatesUsv)):
        outputStatesUsv.append(storedStatesUsv[k][-1])

    trainingCost = costFunction(trainingDataSv, outputStatesSv)
    testingCostAll = costFunction(trainingData, outputStates)
    # testingCostUsv = costFunction(trainingDataUsv, outputStatesUsv)
    testingCostUsv = 1
    if (N - S) != 0:
        testingCostUsv = N / (N - S) * (testingCostAll - S / (N) * trainingCost)
    plotlist = [[s], [trainingCost], [testingCostAll], [testingCostUsv]]

    # Optional
    runtime = time()

    # Training of the Quantum Neural Network
    for k in range(trainingRounds):
        if alert > 0 and k % alert == 0: print("In training round " + str(k))

        ### UPDATING
        newUnitaries = unitariesCopy(currentUnitaries)

        # Loop over layers:
        for l in range(1, len(qnnArch)):
            numInputQubits = qnnArch[l - 1]
            numOutputQubits = qnnArch[l]

            # Loop over perceptrons
            for j in range(numOutputQubits):
                newUnitaries[l][j] = (
                        makeUpdateMatrixTensored(qnnArch, currentUnitaries, trainingDataSv, storedStatesSv, lda, ep, l,
                                                 j) * currentUnitaries[l][j])

        ### FEEDFORWARD
        # Feedforward for given unitaries
        s = s + ep
        currentUnitaries = newUnitaries

        storedStates = feedforward(qnnArch, currentUnitaries, trainingData)
        storedStatesSv = []
        for index in listSv:
            storedStatesSv.append(storedStates[index])

        storedStatesUsv = []
        for index in listUsv:
            storedStatesUsv.append(storedStates[index])

        # Cost calculation for given unitaries
        outputStatesSv = []
        for m in range(len(storedStatesSv)):
            outputStatesSv.append(storedStatesSv[m][-1])

        outputStates = []
        for m in range(len(storedStates)):
            outputStates.append(storedStates[m][-1])

        outputStatesUsv = []
        for k in range(len(storedStatesUsv)):
            outputStatesUsv.append(storedStatesUsv[k][-1])

        trainingCost = costFunction(trainingDataSv, outputStatesSv)
        testingCostAll = costFunction(trainingData, outputStates)
        # testingCostUsv = costFunction(trainingDataUsv, outputStatesUsv)
        testingCostUsv = 1
        if (N - S) != 0:
            testingCostUsv = N / (N - S) * (testingCostAll - S / (N) * trainingCost)
        plotlist[0].append(s)
        plotlist[1].append(trainingCost)
        plotlist[2].append(testingCostAll)
        plotlist[3].append(testingCostUsv)

    # Optional
    runtime = time() - runtime
    print("Trained semisupervised " + str(trainingRounds) + " rounds for a " + str(qnnArch) + " network and " + str(
        len(trainingDataSv)) + " of " + str(len(trainingData)) + " supervised training pairs in " + str(
        round(runtime, 2)) + " seconds")

    # Return
    return [plotlist, currentUnitaries]


# brings the update matrix in the right form to apply on the unitaries
def makeUpdateMatrixTensoredSv(updateMatrix, qnnArch, l, m):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    if numOutputQubits - 1 != 0:
        updateMatrix = qt.tensor(updateMatrix, tensoredId(numOutputQubits - 1))

    # Return
    return swappedOp(updateMatrix, numInputQubits, numInputQubits + m)

# semisupervised QNN, outputs plots and CSV
def mainSsv(qnnArch, numTrainingPairs, numberSupervisedPairs, lda, ep, trainingRounds,  delta, kind, alert=0):
    qnnArchString = ''
    for i in qnnArch:
        # append strings to the variable
        qnnArchString += '-' + str(i)
    qnnArchString = qnnArchString[1:]
    if not os.path.exists(kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_data' + '.txt'):
        print('File does not exist.')
        return None
    # the number of supervised pairs has to be smaller or equal to the total number of pairs
    if numTrainingPairs < numberSupervisedPairs:
        sys.exit("Error: numTrainingPairs < numberSupervisedPairs")

    # creates network and initial unitaries
    network = randomNetwork(qnnArch, numTrainingPairs)  # TODO, not efficient
    initialUnitaries = network[1]

    with open(kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_data' + '.txt', "rb") as fp:  # Unpickling
        dataList = pickle.load(fp)  # =[trainingData, adjMatrix, listSv, trainingDataSv, listUsv, trainingDataUsv]
    print('File ' + kind + '_' + str(numTrainingPairs) + 'pairs' + str(
        numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_data' + '.txt')
    # with dataList=[trainingData, listSv, trainingDataSv, listUsv, trainingDataUsv,  # save the data
    trainingData = dataList[0]
    # (trainingData)
    listSv = dataList[1]
    trainingDataSv = dataList[2]
    listUsv = dataList[3]
    trainingDataUsv = dataList[4]
    fidMatrix = dataList[5]
    # Create training data pairs with random part (uses delta)
    for i in range(0, len(trainingData)):
        trainingData[i][1] = (1 - delta) * trainingData[i][1] + delta * randomQubitState(qnnArch[-1])
        trainingData[i][1] = (1 / sc.linalg.norm(trainingData[i][1])) * trainingData[i][1]
    for i in range(0, len(trainingDataSv)):
        trainingDataSv[i][1] = (1 - delta) * trainingDataSv[i][1] + delta * randomQubitState(qnnArch[-1])
        trainingDataSv[i][1] = (1 / sc.linalg.norm(trainingDataSv[i][1])) * trainingDataSv[i][1]
    for i in range(0, len(trainingDataUsv)):
        trainingDataUsv[i][1] = (1 - delta) * trainingDataUsv[i][1] + delta * randomQubitState(qnnArch[-1])
        trainingDataUsv[i][1] = (1 / sc.linalg.norm(trainingDataUsv[i][1])) * trainingDataUsv[i][1]

    # train ssv structure
    plotlistSsv = \
        qnnTrainingSsv(qnnArch, initialUnitaries, trainingData, trainingDataSv, listSv, trainingDataUsv, listUsv, lda,
                       ep,
                       trainingRounds, alert)[0]

    for i in range(len(plotlistSsv[1])):
        if plotlistSsv[1][i] >= 0.95:
            print("Semisupervised: Exceeds cost of 0.95 at training step " + str(i))
            break

    plt.plot(plotlistSsv[0], plotlistSsv[1], label='QNN Ssv (training)', color='green')
    plt.plot(plotlistSsv[0], plotlistSsv[3], label='QNN Ssv (testing USV)', color='blue')
    plt.xlabel("s * epsilon")
    plt.ylabel("Cost[s]")
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='lower right',
               title=qnnArchString + ',delta =' + str(delta))
    # plt.show()
    df = pd.DataFrame({'step times epsilon': plotlistSsv[0], 'SsvTraining': plotlistSsv[1],
                       'SsvTestingAll': plotlistSsv[2], 'SsvTestingUsv': plotlistSsv[3]})

    # saves plot as figure and csv
    plt.savefig(
        kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_delta' + str(delta).replace('.', 'i') + '_lda' + str(lda).replace('.',
                                                                                                           'i') + '_ep' + str(
            ep).replace('.', 'i') + '_plot.png', bbox_inches='tight', dpi=150)
    plt.clf()
    df.to_csv(
        kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_delta' + str(delta).replace('.', 'i') + '_lda' + str(lda).replace('.',
                                                                                                           'i') + '_ep' + str(
            ep).replace('.', 'i') + '_plot.csv', index=False)


def fidelityMatrixRandomUnitary(qnnArch, numTrainingPairs):
    kind = "randomUnitary"
    # make a unitary for the training data
    networkUnitary = randomQubitUnitary(qnnArch[-1])
    # create the training data
    trainingData = randomTrainingData(networkUnitary, numTrainingPairs)
    # create a sublist of the training data with the (un)supervised pairs (trainingDataSv and trainingDataUsv)
    # and its labels (listSv and listUsv)

    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p

    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixLineUnitary(qnnArch, numTrainingPairs):
    kind = "lineUnitary"
    if numTrainingPairs > 2 ** qnnArch[-1]:
        print('So many orthogonal states do not exist.')
        return None
    networkUnitary = randomQubitUnitary(qnnArch[-1])

    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = qt.basis(2 ** qnnArch[-1], i)
        # t = qt.Qobj(t)
        # Make dims list
        dims1 = [2 for i in range(qnnArch[-1])]
        dims2 = [1 for i in range(qnnArch[-1])]
        dims = [dims1, dims2]
        t.dims = dims
        trainingDataInput.append(t)
    trainingDataInputConnected = []
    lineIndex = list(range(0, numTrainingPairs))
    for i in range(numTrainingPairs):  # -1 when its not a circle
        t = ((lineIndex[-i - 1]) / (numTrainingPairs - 1)) * trainingDataInput[0] + (
                (lineIndex[i]) / (numTrainingPairs - 1)) * trainingDataInput[1]
        t = (1 / sc.linalg.norm(t)) * t
        trainingDataInputConnected.append(t)
    # trainingDataInputConnected.append(trainingDataInput[numTrainingPairs - 1])  # because its not a circle
    trainingData = []
    for i in range(numTrainingPairs):
        t = trainingDataInputConnected[i]
        ut = networkUnitary * t
        trainingData.append([t, ut])
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p
    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixCircleUnitary(qnnArch, numTrainingPairs):
    kind = "circleUnitary"
    if numTrainingPairs > 2 ** qnnArch[-1]:
        print('So many orthogonal states do not exist.')
        return None
    networkUnitary = randomQubitUnitary(qnnArch[-1])

    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = qt.basis(2 ** qnnArch[-1], i)
        # t = qt.Qobj(t)
        # Make dims list
        dims1 = [2 for i in range(qnnArch[-1])]
        dims2 = [1 for i in range(qnnArch[-1])]
        dims = [dims1, dims2]
        t.dims = dims
        trainingDataInput.append(t)
    trainingDataInputConnected = []
    for i in range(numTrainingPairs):  # -1 when its not a circle
        t = trainingDataInput[i] + trainingDataInput[(i + 1) % numTrainingPairs]
        t = (1 / sc.linalg.norm(t)) * t
        trainingDataInputConnected.append(t)
    # trainingDataInputConnected.append(trainingDataInput[numTrainingPairs - 1])  # because its not a circle

    trainingData = []
    for i in range(numTrainingPairs):
        t = trainingDataInputConnected[i]
        ut = networkUnitary * t
        trainingData.append([t, ut])
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p

    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixCircleOutput(qnnArch, numTrainingPairs):
    kind = "circleOutput"
    if numTrainingPairs > 2 ** qnnArch[-1]:
        print('So many orthogonal states do not exist.')
        return None

    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = qt.basis(2 ** qnnArch[-1], i)
        # t = qt.Qobj(t)
        # Make dims list
        dims1 = [2 for i in range(qnnArch[-1])]
        dims2 = [1 for i in range(qnnArch[-1])]
        dims = [dims1, dims2]
        t.dims = dims
        trainingDataInput.append(t)
    trainingDataInputConnected = []
    for i in range(numTrainingPairs):  # -1 when its not a circle
        t = trainingDataInput[i] + trainingDataInput[(i + 1) % numTrainingPairs]
        t = (1 / sc.linalg.norm(t)) * t
        trainingDataInputConnected.append(t)
    # trainingDataInputConnected.append(trainingDataInput[numTrainingPairs - 1])  # because its not a circle

    trainingData = []
    for i in range(numTrainingPairs):
        trainingData.append([trainingDataInput[i], trainingDataInputConnected[i]])
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p
    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixLineOutput(qnnArch, numTrainingPairs):
    kind = "lineOutput"
    if numTrainingPairs > 2 ** qnnArch[-1]:
        print('So many orthogonal states do not exist.')
        return None

    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = qt.basis(2 ** qnnArch[-1], i)
        # t = qt.Qobj(t)
        # Make dims list
        dims1 = [2 for i in range(qnnArch[-1])]
        dims2 = [1 for i in range(qnnArch[-1])]
        dims = [dims1, dims2]
        t.dims = dims
        trainingDataInput.append(t)
    trainingDataInputConnected = []
    lineIndex = list(range(0, numTrainingPairs))
    for i in range(numTrainingPairs):  # -1 when its not a circle
        t = ((lineIndex[-i - 1]) / (numTrainingPairs - 1)) * trainingDataInput[0] + (
                (lineIndex[i]) / (numTrainingPairs - 1)) * trainingDataInput[1]
        t = (1 / sc.linalg.norm(t)) * t
        trainingDataInputConnected.append(t)
    # trainingDataInputConnected.append(trainingDataInput[numTrainingPairs - 1])  # because its not a circle

    trainingData = []
    for i in range(numTrainingPairs):
        trainingData.append([trainingDataInput[i], trainingDataInputConnected[i]])
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p
    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixADC(qnnArch, numTrainingPairs):
    kind = "ADC"
    if numTrainingPairs > 2 ** qnnArch[-1]:
        print('So many orthogonal states do not exist.')
        return None

    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = qt.basis(2 ** qnnArch[-1], i)
        # t = qt.Qobj(t)
        # Make dims list
        dims1 = [2 for i in range(qnnArch[-1])]
        dims2 = [1 for i in range(qnnArch[-1])]
        dims = [dims1, dims2]
        t.dims = dims
        trainingDataInput.append(t)
    trainingDataInputConnected = []
    for i in range(numTrainingPairs):  # -1 when its not a circle
        t = trainingDataInput[i] + trainingDataInput[(i + 1) % numTrainingPairs]
        t = (1 / sc.linalg.norm(t)) * t
        trainingDataInputConnected.append(t)

    trainingData = []
    for i in range(numTrainingPairs):
        t = trainingDataInput[i]
        t2 = trainingDataInputConnected[i]
        trainingData.append([t, t2])
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p

    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixClassification(qnnArch, numTrainingPairs):
    kind = "classification"
    if numTrainingPairs > 2 ** qnnArch[0]:
        print('So many orthogonal states do not exist.')
        return None

    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = qt.basis(2 ** qnnArch[0], i)
        # t = qt.Qobj(t)
        # Make dims list
        dims1 = [2 for i in range(qnnArch[0])]
        dims2 = [1 for i in range(qnnArch[0])]
        dims = [dims1, dims2]
        t.dims = dims
        trainingDataInput.append(t)
    trainingDataInputClassic = []
    for i in range(numTrainingPairs):  # -1 when its not a circle
        t = qt.basis(2 ** 1, 1)
        if i % 2 == 0:
            t = qt.basis(2 ** 1, 0)
        trainingDataInputClassic.append(t)

    trainingData = []
    for i in range(numTrainingPairs):
        t = trainingDataInput[i]
        t2 = trainingDataInputClassic[i]
        trainingData.append([t, t2])
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p

    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixClassificationSuperposition(qnnArch, numTrainingPairs):
    kind = "classificationSup"
    if numTrainingPairs > 2 ** qnnArch[0]:
        print('So many orthogonal states do not exist.')
        return None

    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = qt.basis(2 ** qnnArch[0], i)
        # t = qt.Qobj(t)
        # Make dims list
        dims1 = [2 for i in range(qnnArch[0])]
        dims2 = [1 for i in range(qnnArch[0])]
        dims = [dims1, dims2]
        t.dims = dims
        trainingDataInput.append(t)
    trainingDataInputClassic = []
    for i in range(numTrainingPairs):  # -1 when its not a circle
        t = qt.basis(2 ** 1, 1)
        if i % 2 == 0:
            t = qt.basis(2 ** 1, 0)
        trainingDataInputClassic.append(t)

    trainingData = []
    for i in range(numTrainingPairs):
        t = trainingDataInput[i]
        t2 = trainingDataInputClassic[i]
        trainingData.append([t, t2])
    sup = qt.basis(2 ** 1, 0) + qt.basis(2 ** 1, 1)
    sup = (1 / sc.linalg.norm(sup)) * sup
    trainingData[-1][1] = sup
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p
    # !!! to connect the superposition with one of each cluster
    fidMatrix[-1][0] = 1
    fidMatrix[0][-1] = 1
    fidMatrix[-1][1] = 1
    fidMatrix[1][-1] = 1
    print(fidMatrix)
    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixLineOutputRandom(qnnArch, numTrainingPairs):
    kind = "lineOutputRandom"
    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = randomQubitState(qnnArch[0])
        trainingDataInput.append(t)
    trainingDataOutput = []
    lineIndex = list(range(0, numTrainingPairs))
    for i in range(numTrainingPairs):  # -1 when its not a circle
        t = ((lineIndex[-i - 1]) / (numTrainingPairs - 1)) * qt.basis(2 ** qnnArch[-1], 0) + (
                (lineIndex[i]) / (numTrainingPairs - 1)) * qt.basis(2 ** qnnArch[-1], 1)
        t = (1 / sc.linalg.norm(t)) * t
        trainingDataOutput.append(t)
    # trainingDataInputConnected.append(trainingDataInput[numTrainingPairs - 1])  # because its not a circle
    trainingData = []
    for i in range(numTrainingPairs):
        trainingData.append([trainingDataInput[i], trainingDataOutput[i]])
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p
    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixLineOutputRandomShuffled(qnnArch, numTrainingPairs):
    kind = "lineOutputRandomShuffled"
    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = randomQubitState(qnnArch[0])
        trainingDataInput.append(t)
    trainingDataOutput = []
    lineIndex = list(range(0, numTrainingPairs))
    for i in range(numTrainingPairs):  # -1 when its not a circle
        t = ((lineIndex[-i - 1]) / (numTrainingPairs - 1)) * qt.basis(2 ** qnnArch[-1], 0) + (
                (lineIndex[i]) / (numTrainingPairs - 1)) * qt.basis(2 ** qnnArch[-1], 1)
        t = (1 / sc.linalg.norm(t)) * t
        trainingDataOutput.append(t)
    # trainingDataInputConnected.append(trainingDataInput[numTrainingPairs - 1])  # because its not a circle
    trainingData = []
    for i in range(numTrainingPairs):
        trainingData.append([trainingDataInput[i], trainingDataOutput[i]])
    random.shuffle(trainingData)
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p
    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixClustersRandom(qnnArch, numTrainingPairs):
    kind = "clustersRandom"
    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = randomQubitState(qnnArch[0])
        trainingDataInput.append(t)
    trainingDataOutput = []
    width = 2
    lineIndex = list(range(0, numTrainingPairs * width))
    for i in range(numTrainingPairs * width):  # -1 when its not a circle
        t = ((lineIndex[-i - 1]) / (numTrainingPairs * width - 1)) * qt.basis(2 ** qnnArch[-1], 0) + (
                (lineIndex[i]) / (numTrainingPairs * width - 1)) * qt.basis(2 ** qnnArch[-1], 1)
        t = (1 / sc.linalg.norm(t)) * t
        trainingDataOutput.append(t)
    while len(trainingDataOutput) != numTrainingPairs:
        trainingDataOutput.pop(int(len(trainingDataOutput) / 2))
    # trainingDataInputConnected.append(trainingDataInput[numTrainingPairs - 1])  # because its not a circle
    trainingData = []
    for i in range(numTrainingPairs):
        trainingData.append([trainingDataInput[i], trainingDataOutput[i]])
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p
    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixClustersRandomShuffled(qnnArch, numTrainingPairs):
    kind = "clustersRandomShuffled"
    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = randomQubitState(qnnArch[0])
        trainingDataInput.append(t)
    trainingDataOutput = []
    width = 2
    lineIndex = list(range(0, numTrainingPairs * width))
    for i in range(numTrainingPairs * width):  # -1 when its not a circle
        t = ((lineIndex[-i - 1]) / (numTrainingPairs * width - 1)) * qt.basis(2 ** qnnArch[-1], 0) + (
                (lineIndex[i]) / (numTrainingPairs * width - 1)) * qt.basis(2 ** qnnArch[-1], 1)
        t = (1 / sc.linalg.norm(t)) * t
        trainingDataOutput.append(t)
    while len(trainingDataOutput) != numTrainingPairs:
        trainingDataOutput.pop(int(len(trainingDataOutput) / 2))
    # trainingDataInputConnected.append(trainingDataInput[numTrainingPairs - 1])  # because its not a circle
    trainingData = []
    for i in range(numTrainingPairs):
        trainingData.append([trainingDataInput[i], trainingDataOutput[i]])
    random.shuffle(trainingData)
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p
    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixConnectedClustersRandom(qnnArch, numTrainingPairs):
    kind = "connectedClustersRandom"
    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = randomQubitState(qnnArch[0])
        trainingDataInput.append(t)
    trainingDataOutput = []
    width = 2
    lineIndex = list(range(0, numTrainingPairs * width))
    for i in range(numTrainingPairs * width):  # -1 when its not a circle
        t = ((lineIndex[-i - 1]) / (numTrainingPairs * width - 1)) * qt.basis(2 ** qnnArch[-1], 0) + (
                (lineIndex[i]) / (numTrainingPairs * width - 1)) * qt.basis(2 ** qnnArch[-1], 1)
        t = (1 / sc.linalg.norm(t)) * t
        trainingDataOutput.append(t)
    print(int(len(trainingDataOutput) / 2))
    connectionstate = trainingDataOutput.pop(int(len(trainingDataOutput) / 2))
    while len(trainingDataOutput) != numTrainingPairs - 1:
        trainingDataOutput.pop(int(len(trainingDataOutput) / 2))
    trainingDataOutput.append(connectionstate)
    trainingData = []
    for i in range(numTrainingPairs):
        trainingData.append([trainingDataInput[i], trainingDataOutput[i]])
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p
    print(fidMatrix)
    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


def fidelityMatrixConnectedClustersRandomShuffled(qnnArch, numTrainingPairs):
    kind = "connectedClustersRandomShuffled"
    trainingDataInput = []
    for i in range(numTrainingPairs):
        t = randomQubitState(qnnArch[0])
        trainingDataInput.append(t)
    trainingDataOutput = []
    width = 2
    lineIndex = list(range(0, numTrainingPairs * width))
    for i in range(numTrainingPairs * width):  # -1 when its not a circle
        t = ((lineIndex[-i - 1]) / (numTrainingPairs * width - 1)) * qt.basis(2 ** qnnArch[-1], 0) + (
                (lineIndex[i]) / (numTrainingPairs * width - 1)) * qt.basis(2 ** qnnArch[-1], 1)
        t = (1 / sc.linalg.norm(t)) * t
        trainingDataOutput.append(t)
    connectionstate = trainingDataOutput.pop(int(len(trainingDataOutput) / 2))
    while len(trainingDataOutput) != numTrainingPairs - 1:
        trainingDataOutput.pop(int(len(trainingDataOutput) / 2))
    trainingDataOutput.append(connectionstate)
    trainingData = []
    for i in range(numTrainingPairs):
        trainingData.append([trainingDataInput[i], trainingDataOutput[i]])
    random.shuffle(trainingData)
    fidMatrix = np.identity(numTrainingPairs)
    for i in range(0, numTrainingPairs):
        for j in range(0, i):
            p = trainingData[i][1].dag() * (trainingData[j][1] * trainingData[j][1].dag()) * trainingData[i][1]
            fidMatrix[i][j] = p.tr()
            fidMatrix[j][i] = p.tr()
            # oder
            # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
            # fidMatrix[i][j] = p
    print(fidMatrix)
    # make fidelity plot
    plt.matshow(fidMatrix);
    plt.colorbar()
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.png',
                bbox_inches='tight',
                dpi=150)
    plt.clf()
    fidelityMatrixAndTrainingData = [fidMatrix, trainingData]
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "wb") as fp:  # Pickling
        pickle.dump(fidelityMatrixAndTrainingData, fp)


### main functions

# saves all needed data
def listSv(qnnArch, numTrainingPairs, supervisedPairsList, kind):
    # the number of supervised pairs has to be smaller or equal to the total number of pairs
    numSupervisedPairs = len(supervisedPairsList)
    # supervisedPairsList does not inlude 0, is starting for example with 1
    listSv = []
    for element in supervisedPairsList:
        listSv.append(element - 1)
    if numTrainingPairs < numSupervisedPairs:
        print('Error: numTrainingPairs < numSupervisedPairs')
        return None
    if not os.path.exists(
            kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt'):
        print('File does not exist.')
        return None
    with open(kind + '_' + str(numTrainingPairs) + 'pairs_' + str(qnnArch[-1]) + 'qubit_fidelities' + '.txt',
              "rb") as fp:  # Unpickling
        fidelityMatrixAndTrainingData = pickle.load(fp)
    fidMatrix = fidelityMatrixAndTrainingData[0]
    trainingData = fidelityMatrixAndTrainingData[1]
    listNumTrainingPairs = range(0, numTrainingPairs)
    listUsv = list(range(0, numTrainingPairs))  # indices of unsupervised
    for i in listSv:
        listUsv.remove(i)
    trainingDataSv = []
    for index in listSv:
        trainingDataSv.append(trainingData[index])
    trainingDataUsv = []
    for index in listUsv:
        trainingDataUsv.append(trainingData[index])

    qnnArchString = ''
    for i in qnnArch:
        # append strings to the variable
        qnnArchString += '-' + str(i)
    qnnArchString = qnnArchString[1:]

    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    dataList = [trainingData, listSv, trainingDataSv, listUsv,
                 trainingDataUsv, fidMatrix]  # save the data
    with open(kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numSupervisedPairs) + 'sv_' + qnnArchString + 'network_data' + '.txt', "wb") as fp:  # Pickling
        pickle.dump(dataList, fp)

# plots and saves the losses with number of supervised pairs on the x-axis
def makeLossListIndex(qnnArch, numTrainingPairs, numberSupervisedPairsList, lda, ep, trainingRounds, delta, index, kind):
    # load dataframe from csv
    SsvTraining = []
    SsvTestingAll = []
    SsvTestingUsv = []
    qnnArchString = ''
    for i in qnnArch:
        # append strings to the variable
        qnnArchString += '-' + str(i)
    qnnArchString = qnnArchString[1:]
    for numberSupervisedPairs in numberSupervisedPairsList:
        readdf = pd.read_csv(kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_delta' + str(delta).replace('.',
                                                                                                                   'i') + '_lda' + str(
            lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_plot.csv')
        SsvTraining.append(float(readdf.tail(1)['SsvTraining']))
        SsvTestingAll.append(float(readdf.tail(1)['SsvTestingAll']))
        SsvTestingUsv.append(float(readdf.tail(1)['SsvTestingUsv']))

    plt.scatter(numberSupervisedPairsList, SsvTraining, label='QNN Ssv (training)', color='green')
    plt.scatter(numberSupervisedPairsList, SsvTestingUsv, label='QNN Ssv (testing USV)', color='blue')
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='lower right',
               title=qnnArchString + ',  delta =' + str(delta))
    plt.savefig(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + qnnArchString + 'network_delta' + str(delta).replace(
            '.', 'i') + '_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_plot' + str(
            index) + '.png', bbox_inches='tight', dpi=150)
    plt.clf()
    df = pd.DataFrame(
        {'numberSupervisedPairsList': numberSupervisedPairsList, 'SsvTraining': SsvTraining,
         'SsvTestingAll': SsvTestingAll, 'SsvTestingUsv': SsvTestingUsv})
    df.to_csv(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + qnnArchString + 'network_delta' + str(delta).replace(
            '.', 'i') + '_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_plot' + str(
            index) + '.csv', index=False)


# main function for QNN, training loss and testing loss and makes CSV file
# plots and saves the losses with number of supervised pairs on the x-axis averaged for number of shots
def makeLossListMean(qnnArch, numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, delta, shots, kind):
    numberSupervisedPairsList = range(1, maxNumberSupervisedPairs + 1)
    for shotindex in range(1, shots + 1):
        # prints number of shot
        print('------ shot ' + str(shotindex) + ' of ' + str(shots) + ' ------')
        # finds the right filelity matrix function for the kind of data and checks if enough orthogonal states
        if kind == "randomUnitary":
            if qnnArch[-1] != qnnArch[0]:
                print('The last layer has to be of the same size as the first layer.')
                return None
            fidelityMatrixRandomUnitary(qnnArch, numTrainingPairs)
        elif kind == "circleUnitary":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            fidelityMatrixCircleUnitary(qnnArch, numTrainingPairs)
        elif kind == "lineUnitary":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            fidelityMatrixLineUnitary(qnnArch, numTrainingPairs)
        elif kind == "ADC":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            elif qnnArch[-1] != qnnArch[0]:
                print('The last layer has to be of the same size as the first layer.')
                return None
            fidelityMatrixADC(qnnArch, numTrainingPairs)
        elif kind == "classification":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            elif qnnArch[-1] != 1:
                print('The last layer has to be a one qubit layer.')
                return None
            fidelityMatrixClassification(qnnArch, numTrainingPairs)
        elif kind == "classificationSup":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            elif qnnArch[-1] != 1:
                print('The last layer has to be a one qubit layer.')
                return None
            fidelityMatrixClassificationSuperposition(qnnArch, numTrainingPairs)
        elif kind == "circleOutput":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            fidelityMatrixCircleOutput(qnnArch, numTrainingPairs)
        elif kind == "lineOutput":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            fidelityMatrixLineOutput(qnnArch, numTrainingPairs)
        elif kind == "lineOutputRandom":
            fidelityMatrixLineOutputRandom(qnnArch, numTrainingPairs)
        elif kind == "lineOutputRandomShuffled":
            fidelityMatrixLineOutputRandomShuffled(qnnArch, numTrainingPairs)
        elif kind == "clustersRandom":
            fidelityMatrixClustersRandom(qnnArch, numTrainingPairs)
        elif kind == "clustersRandomShuffled":
            fidelityMatrixClustersRandomShuffled(qnnArch, numTrainingPairs)
        elif kind == "connectedClustersRandom":
            fidelityMatrixConnectedClustersRandom(qnnArch, numTrainingPairs)
        elif kind == "connectedClustersRandomShuffled":
            fidelityMatrixConnectedClustersRandomShuffled(qnnArch, numTrainingPairs)
        else:
            print("Kind does not exist.")
            return None
        numberSupervisedPairsListData = numberSupervisedPairsList
        for i in numberSupervisedPairsList:
            listSv(qnnArch, numTrainingPairs, numberSupervisedPairsListData, kind)
            numberSupervisedPairsListData = numberSupervisedPairsListData[:-1]

        for numberSupervisedPairs in numberSupervisedPairsList:
            mainSsv(qnnArch, numTrainingPairs, numberSupervisedPairs, lda, ep, trainingRounds, delta, kind)

        makeLossListIndex(qnnArch, numTrainingPairs, numberSupervisedPairsList, lda, ep, trainingRounds, delta, shotindex, kind)
    # load dataframe from csv
    SsvTrainingMeanList = []
    SsvTestingAllMeanList = []
    SsvTestingUsvMeanList = []
    for indexSv in numberSupervisedPairsList:
        SsvTraining = []
        SsvTestingAll = []
        SsvTestingUsv = []
        qnnArchString = ''
        for i in qnnArch:
            # append strings to the variable
            qnnArchString += '-' + str(i)
        qnnArchString = qnnArchString[1:]
        for index in range(1, shots + 1):
            readdf = pd.read_csv(
                kind + '_' + str(numTrainingPairs) + 'pairs_' + qnnArchString + 'network_delta' + str(
                    delta).replace(
                    '.', 'i') + '_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.',
                                                                                              'i') + '_rounds' + str(
                    trainingRounds) + '_plot' + str(
                    index) + '.csv')
            SsvTraining.append(readdf._get_value(indexSv - 1, 'SsvTraining'))
            SsvTestingAll.append(readdf._get_value(indexSv - 1, 'SsvTestingAll'))
            SsvTestingUsv.append(readdf._get_value(indexSv - 1, 'SsvTestingUsv'))
        # means
        SsvTrainingMean = sum(SsvTraining) / len(SsvTraining)
        SsvTestingAllMean = sum(SsvTestingAll) / len(SsvTestingAll)
        SsvTestingUsvMean = sum(SsvTestingUsv) / len(SsvTestingUsv)
        # collect means in list
        SsvTrainingMeanList.append(SsvTrainingMean)
        SsvTestingAllMeanList.append(SsvTestingAllMean)
        SsvTestingUsvMeanList.append(SsvTestingUsvMean)
    # plots
    # or add plot SsvTrainingMean
    plt.scatter(numberSupervisedPairsList, SsvTrainingMeanList, label='QNN Ssv (training)', color='green')
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='lower right',
               title=qnnArchString + ', delta =' + str(delta))
    plt.scatter(numberSupervisedPairsList, SsvTestingUsvMeanList, label='QNN Ssv (testing USV)', color='blue')

    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='lower right',
               title=qnnArchString + ', delta =' + str(delta))
    plt.savefig(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + qnnArchString + 'network_delta' + str(
            delta).replace(
            '.', 'i') + '_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_shots' + str(shots) + '_plotmean.png',
        bbox_inches='tight', dpi=150)
    plt.clf()
    df = pd.DataFrame(
        {'numberSupervisedPairsList': numberSupervisedPairsList, 'SsvTrainingMeanList': SsvTrainingMeanList,
         'SsvTestingAllMeanList': SsvTestingAllMeanList, 'SsvTestingUsvMeanList': SsvTestingUsvMeanList})
    df.to_csv(kind + '_' +
              str(numTrainingPairs) + 'pairs_' + qnnArchString + 'network_delta' + str(delta).replace(
        '.', 'i') + '_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
        trainingRounds) + '_shots' + str(shots) + '_plotmean.csv',
              index=False)


def makeLossListIndexDelta(qnnArch, numTrainingPairs, numberSupervisedPairs, lda, ep, trainingRounds, deltaList, index, kind):
    # load dataframe from csv
    SsvTraining = []
    SsvTestingAll = []
    SsvTestingUsv = []
    qnnArchString = ''
    for i in qnnArch:
        # append strings to the variable
        qnnArchString += '-' + str(i)
    qnnArchString = qnnArchString[1:]
    for delta in deltaList:
        readdf = pd.read_csv(kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_delta' + str(delta).replace('.','i') + '_lda' + str(
            lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_plot.csv')
        SsvTraining.append(float(readdf.tail(1)['SsvTraining']))
        SsvTestingAll.append(float(readdf.tail(1)['SsvTestingAll']))
        SsvTestingUsv.append(float(readdf.tail(1)['SsvTestingUsv']))

    plt.scatter(deltaList, SsvTraining, label='QNN Ssv (training)', color='green')
    plt.scatter(deltaList, SsvTestingUsv, label='QNN Ssv (testing USV)', color='blue')
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='lower right',
               title=qnnArchString + ',  numberTrainingPairs =' + str(numTrainingPairs)+ ',  numberSupervisedPairs =' + str(numberSupervisedPairs))
    plt.savefig(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + str(numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_deltaVar_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_plot' + str(
            index) + '.png', bbox_inches='tight', dpi=150)
    plt.clf()
    df = pd.DataFrame(
        {'deltaList': deltaList, 'SsvTraining': SsvTraining,
         'SsvTestingAll': SsvTestingAll, 'SsvTestingUsv': SsvTestingUsv})
    df.to_csv(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + str(numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_deltaVar_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_plot' + str(
            index) + '.csv', index=False)

def makeLossListMeanDelta(qnnArch, numTrainingPairs, numberSupervisedPairs, lda, ep, trainingRounds, deltaList,shots, kind):
    numberSupervisedPairsList = range(1, numberSupervisedPairs + 1)
    for shotindex in range(1, shots + 1):
        # prints number of shot
        print('------ shot ' + str(shotindex) + ' of ' + str(shots) + ' ------')
        # finds the right filelity matrix function for the kind of data and checks if enough orthogonal states
        if kind == "randomUnitary":
            if qnnArch[-1] != qnnArch[0]:
                print('The last layer has to be of the same size as the first layer.')
                return None
            fidelityMatrixRandomUnitary(qnnArch, numTrainingPairs)
        elif kind == "circleUnitary":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            fidelityMatrixCircleUnitary(qnnArch, numTrainingPairs)
        elif kind == "lineUnitary":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            fidelityMatrixLineUnitary(qnnArch, numTrainingPairs)
        elif kind == "ADC":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            elif qnnArch[-1] != qnnArch[0]:
                print('The last layer has to be of the same size as the first layer.')
                return None
            fidelityMatrixADC(qnnArch, numTrainingPairs)
        elif kind == "classification":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            elif qnnArch[-1] != 1:
                print('The last layer has to be a one qubit layer.')
                return None
            fidelityMatrixClassification(qnnArch, numTrainingPairs)
        elif kind == "classificationSup":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            elif qnnArch[-1] != 1:
                print('The last layer has to be a one qubit layer.')
                return None
            fidelityMatrixClassificationSuperposition(qnnArch, numTrainingPairs)
        elif kind == "circleOutput":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            fidelityMatrixCircleOutput(qnnArch, numTrainingPairs)
        elif kind == "lineOutput":
            if numTrainingPairs > 2 ** qnnArch[0]:
                print('So many orthogonal states do not exist.')
                return None
            fidelityMatrixLineOutput(qnnArch, numTrainingPairs)
        elif kind == "lineOutputRandom":
            fidelityMatrixLineOutputRandom(qnnArch, numTrainingPairs)
        elif kind == "lineOutputRandomShuffled":
            fidelityMatrixLineOutputRandomShuffled(qnnArch, numTrainingPairs)
        elif kind == "clustersRandom":
            fidelityMatrixClustersRandom(qnnArch, numTrainingPairs)
        elif kind == "clustersRandomShuffled":
            fidelityMatrixClustersRandomShuffled(qnnArch, numTrainingPairs)
        elif kind == "connectedClustersRandom":
            fidelityMatrixConnectedClustersRandom(qnnArch, numTrainingPairs)
        elif kind == "connectedClustersRandomShuffled":
            fidelityMatrixConnectedClustersRandomShuffled(qnnArch, numTrainingPairs)
        else:
            print("Kind does not exist.")
            return None
        for delta in deltaList:
            print("delta = " + str(delta))
            listSv(qnnArch, numTrainingPairs, numberSupervisedPairsList, kind)
            mainSsv(qnnArch, numTrainingPairs, numberSupervisedPairs, lda, ep, trainingRounds, delta, kind)

        makeLossListIndexDelta(qnnArch, numTrainingPairs, numberSupervisedPairs, lda, ep, trainingRounds, deltaList, shotindex, kind)

    # load dataframe from csv
    SsvTrainingMeanList = []
    SsvTestingAllMeanList = []
    SsvTestingUsvMeanList = []
    indexDelta = 0
    for element in deltaList:
        SsvTraining = []
        SsvTestingAll = []
        SsvTestingUsv = []
        qnnArchString = ''
        for i in qnnArch:
            # append strings to the variable
            qnnArchString += '-' + str(i)
        qnnArchString = qnnArchString[1:]
        for index in range(1, shots + 1):
            readdf = pd.read_csv(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + str(numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_deltaVar_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_plot' + str(
            index) + '.csv')
            SsvTraining.append(readdf._get_value(indexDelta, 'SsvTraining'))
            SsvTestingAll.append(readdf._get_value(indexDelta, 'SsvTestingAll'))
            SsvTestingUsv.append(readdf._get_value(indexDelta, 'SsvTestingUsv'))
        # means
        SsvTrainingMean = sum(SsvTraining) / len(SsvTraining)
        SsvTestingAllMean = sum(SsvTestingAll) / len(SsvTestingAll)
        SsvTestingUsvMean = sum(SsvTestingUsv) / len(SsvTestingUsv)
        # collect means in list
        SsvTrainingMeanList.append(SsvTrainingMean)
        SsvTestingAllMeanList.append(SsvTestingAllMean)
        SsvTestingUsvMeanList.append(SsvTestingUsvMean)
        indexDelta=indexDelta+1
    # plots
    # or add plot SsvTrainingMean
    plt.scatter(deltaList, SsvTrainingMeanList, label='QNN Ssv (training)', color='green')
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='lower right',
               title=qnnArchString + ',  numberTrainingPairs =' + str(numTrainingPairs)+ ',  numberSupervisedPairs =' + str(numberSupervisedPairs))
    plt.scatter(deltaList, SsvTestingUsvMeanList, label='QNN Ssv (testing USV)', color='blue')

    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='lower right',
               title=qnnArchString + ',  numberTrainingPairs =' + str(numTrainingPairs)+ ',  numberSupervisedPairs =' + str(numberSupervisedPairs))
    plt.savefig(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + str(numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_deltaVar_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_shots' + str(shots) + '_plotmean.png',
        bbox_inches='tight', dpi=150)
    plt.clf()
    df = pd.DataFrame(
        {'deltaList': deltaList, 'SsvTrainingMeanList': SsvTrainingMeanList,
         'SsvTestingAllMeanList': SsvTestingAllMeanList, 'SsvTestingUsvMeanList': SsvTestingUsvMeanList})
    df.to_csv(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + str(numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_deltaVar_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_shots' + str(shots) + '_plotmean.csv',
              index=False)

####### DQNN

numTrainingPairs=100 # number of training pairs
maxNumberSupervisedPairs=20 # the code starts with one supervised training pair and ends with maxNumberSupervisedPairs supervised training pairs
lda = 1 # lambda = 1/learning rate
ep = 0.01 # epsilon = step size
trainingRounds=1000
delta = 0 # set to 0 or use it: creates noise in the training pairs
shots = 10 # number of shots for mean values in the plot
kind = "randomUnitary" # learning a random unitary

### testing the generalisation behaviour
makeLossListMean([1,1,1], numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, delta, shots,kind)
makeLossListMean([1,2,1], numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, delta, shots,kind)
makeLossListMean([2,3,2], numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, delta, shots,kind)
makeLossListMean([3, 4, 3], numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, delta, shots,kind)
makeLossListMean([2, 3, 4, 3, 2], numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, delta, shots,kind)

### testing the noise robustness
deltaList = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
makeLossListMeanDelta([1,1,1], numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, deltaList, shots,kind)
makeLossListMeanDelta([1,2,1], numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, deltaList, shots,kind)
makeLossListMeanDelta([2,3,2], numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, deltaList, shots,kind)
makeLossListMeanDelta([3, 4, 3], numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, deltaList, shots,kind)
makeLossListMeanDelta([2, 3, 4, 3, 2], numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, deltaList, shots,kind)


######  Code for Quantum Generative adversial network

### Package-imports, universal definitions and remarks
import scipy as sc
import qutip as qt
import pandas as pd
from time import time
from random import randint
from random import shuffle
from qutip import Bloch, Qobj
import matplotlib.pyplot as plt
import numpy as np
import pickle


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


### Quantum related helper function

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


##### New stuff for QGAN

def randomNetworkOnly(qnnArch):
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
    return networkUnitaries


def randomStates(numQubits, numTrainingData):
    trainingData = []
    for i in range(numTrainingData):
        t = randomQubitState(numQubits)
        trainingData.append(t)
    return trainingData


def randomStatesMat(numQubits, numTrainingData):  # TODO only for testing, delete later
    trainingData = []
    for i in range(numTrainingData):
        t = randomQubitState(numQubits)
        t = t * t.dag()
        trainingData.append(t)
    return trainingData

def costFunctionGen(outputStates, qnnArch):
    state1 = qt.basis(2 ** qnnArch[-1], 2 ** qnnArch[-1] - 1)
    dims1 = [2 for i in range(qnnArch[-1])]
    dims2 = [1 for i in range(qnnArch[-1])]
    dims = [dims1, dims2]
    state1.dims = dims
    costSum = 0
    if len(outputStates) == 0:  # new
        return 1
    for i in range(len(outputStates)):
        costSum += state1.dag() * outputStates[i] * state1
    return costSum.tr() / len(outputStates)

def costFunctionDis1(outputStates, qnnArch):
    state0 = qt.basis(2 ** qnnArch[-1], 0)
    dims1 = [2 for i in range(qnnArch[-1])]
    dims2 = [1 for i in range(qnnArch[-1])]
    dims = [dims1, dims2]
    state0.dims = dims
    costSum = 0
    if len(outputStates) == 0:  # new
        return 1
    for i in range(len(outputStates)):
        costSum += state0.dag() * outputStates[i] * state0
    return costSum.tr() / len(outputStates)


def costFunctionDis2(outputStates, qnnArch):
    state0 = qt.basis(2 ** qnnArch[-1], 2 ** qnnArch[-1] - 1)
    dims1 = [2 for i in range(qnnArch[-1])]
    dims2 = [1 for i in range(qnnArch[-1])]
    dims = [dims1, dims2]
    state0.dims = dims
    costSum = 0
    if len(outputStates) == 0:  # new
        return 1
    for i in range(len(outputStates)):
        costSum += state0.dag() * outputStates[i] * state0
    return costSum.tr() / len(outputStates)


def feedforwardGAN(qnnArch, unitaries, inputData):
    storedStates = []
    for x in range(len(inputData)):
        currentState = inputData[x] * inputData[x].dag()
        layerwiseList = [currentState]
        for l in range(1, len(qnnArch)):
            currentState = makeLayerChannel(qnnArch, unitaries, l, currentState)
            layerwiseList.append(currentState)
        storedStates.append(layerwiseList)
    return storedStates


def updateMatrixFirstPartGen(qnnArch, qnnArchGen, unitaries, storedStates, l, j, x):
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


def updateMatrixSecondPartGen(qnnArch, qnnArchGen, unitaries, l, j, x,trainingData):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]
    state1 = qt.basis(2 ** qnnArch[-1], 2 ** qnnArch[-1] - 1)
    dims1 = [2 for i in range(qnnArch[-1])]
    dims2 = [1 for i in range(qnnArch[-1])]
    dims = [dims1, dims2]
    state1.dims = dims
    # Calculate sigma state
    state = state1 * state1.dag()
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

def makeUpdateMatrixGen(qnnArch, qnnArchGen, unitaries, storedStates, lda, ep, l, j,trainingData):
    numInputQubits = qnnArch[l - 1]

    # Calculate the sum:
    summ = 0
    for x in range(len(storedStates)):
        # Calculate the commutator
        firstPart = updateMatrixFirstPartGen(qnnArch, qnnArchGen, unitaries, storedStates, l, j, x)
        secondPart = updateMatrixSecondPartGen(qnnArch, qnnArchGen, unitaries, l, j, x,trainingData)
        mat = qt.commutator(firstPart, secondPart)

        # Trace out the rest
        keep = list(range(numInputQubits))
        keep.append(numInputQubits + j)
        mat = partialTraceKeep(mat, keep)

        # Add to sum
        summ = summ + mat

    # Calculate the update matrix from the sum
    summ = (-ep * (2 ** numInputQubits) / (lda * len(storedStates))) * summ
    return summ.expm()

def makeUpdateMatrixTensoredGen(qnnArch, qnnArchGen, unitaries, lda, ep, storedStates, l, j,trainingData):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    res = makeUpdateMatrixGen(qnnArch, qnnArchGen, unitaries, lda, ep, storedStates, l, j,trainingData)
    if numOutputQubits - 1 != 0:
        res = qt.tensor(res, tensoredId(numOutputQubits - 1))
    return swappedOp(res, numInputQubits, numInputQubits + j)

def qnnTrainingGen(qnnArch, qnnArchGen, currentUnitaries, storedStates, lda,ep,trainingData):
    newUnitaries = unitariesCopy(currentUnitaries)
    # Loop over layers:
    for l in range(1, len(qnnArch)):
        numInputQubits = qnnArch[l - 1]
        numOutputQubits = qnnArch[l]

        # Loop over perceptrons
        for j in range(numOutputQubits):
            newUnitaries[l][j] = (makeUpdateMatrixTensoredGen(qnnArch, qnnArchGen, currentUnitaries, storedStates, lda, ep, l,j,trainingData) * currentUnitaries[l][j])
    currentUnitaries = newUnitaries
    return currentUnitaries

def updateMatrixFirstPartDis(qnnArch, qnnArchGen, unitaries, storedStates, storedStatesDis, l, j, x):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]
    # Tensor input state
    state = qt.tensor(storedStates[x][l - 1], tensoredQubit0(numOutputQubits))
    if l >= len(qnnArchGen):
        stateTau = qt.tensor(storedStatesDis[x][l-len(qnnArchGen)], tensoredQubit0(numOutputQubits))
        #print(state.dims, stateTau.dims)
        state=stateTau-state
    # Calculate needed product unitary
    productUni = unitaries[l][0]
    for i in range(1, j + 1):
        productUni = unitaries[l][i] * productUni

    # Multiply
    return productUni * state * productUni.dag()


def updateMatrixSecondPartDis(qnnArch, qnnArchGen, unitaries, l, j, x,trainingData):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]
    state1 = qt.basis(2 ** qnnArch[-1], 2 ** qnnArch[-1] - 1)
    dims1 = [2 for i in range(qnnArch[-1])]
    dims2 = [1 for i in range(qnnArch[-1])]
    dims = [dims1, dims2]
    state1.dims = dims
    # Calculate sigma state
    state = state1 * state1.dag()
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

def makeUpdateMatrixDis(qnnArch, qnnArchGen, unitaries, storedStates, storedStatesDis, lda, ep, l, j,trainingData):
    numInputQubits = qnnArch[l - 1]

    # Calculate the sum:
    summ = 0
    for x in range(len(storedStates)):
        # Calculate the commutator
        firstPart = updateMatrixFirstPartDis(qnnArch, qnnArchGen, unitaries, storedStates, storedStatesDis, l, j, x)
        secondPart = updateMatrixSecondPartDis(qnnArch, qnnArchGen, unitaries, l, j, x,trainingData)
        mat = qt.commutator(firstPart, secondPart)

        # Trace out the rest
        keep = list(range(numInputQubits))
        keep.append(numInputQubits + j)
        mat = partialTraceKeep(mat, keep)

        # Add to sum
        summ = summ + mat

    # Calculate the update matrix from the sum
    summ = (-ep * (2 ** numInputQubits) / (lda * len(storedStates))) * summ
    return summ.expm()

def makeUpdateMatrixTensoredDis(qnnArch, qnnArchGen, unitaries, lda, ep, storedStates, storedStatesDis, l, j,trainingData):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    res = makeUpdateMatrixDis(qnnArch, qnnArchGen, unitaries, lda, ep, storedStates, storedStatesDis, l, j,trainingData)
    if numOutputQubits - 1 != 0:
        res = qt.tensor(res, tensoredId(numOutputQubits - 1))
    return swappedOp(res, numInputQubits, numInputQubits + j)

def qnnTrainingDis(qnnArch, qnnArchGen, currentUnitaries, storedStates, storedStatesDis, lda,ep,trainingData):
    newUnitaries = unitariesCopy(currentUnitaries)
    # Loop over layers:
    for l in range(1, len(qnnArch)):
        numInputQubits = qnnArch[l - 1]
        numOutputQubits = qnnArch[l]

        # Loop over perceptrons
        for j in range(numOutputQubits):
            newUnitaries[l][j] = (makeUpdateMatrixTensoredDis(qnnArch, qnnArchGen, currentUnitaries, storedStates, storedStatesDis, lda, ep, l,j,trainingData) * currentUnitaries[l][j])
    currentUnitaries = newUnitaries
    return currentUnitaries

def generatorTesting(numIn, numTest, qnnArchGen, networkUnitariesGen, trainingData):
    # testing generator
    testIncomes = randomStates(numIn, numTest)
    testOutcomes = feedforwardGAN(qnnArchGen, networkUnitariesGen, testIncomes)
    outputStates = []
    for i in range(len(testOutcomes)):
        outputStates.append(testOutcomes[i][-1])
        """
        peakMaxList = []
        for trainingState in trainingData:
            for outputState in outputStates:
                peakMaxList.append((trainingState.dag() * outputState * trainingState).tr())
        return max(peakMaxList)
        """
        # I think the previous part should be substituted by
        peakMaxList = []
        for outputState in outputStates:
            all_fidelities = [(trainingState.dag() * outputState * trainingState).tr() for trainingState in
                              trainingData]
            peakMaxList.append(max(all_fidelities))
    return np.average(peakMaxList)

def costFunctionHilbertTest(currentOutput):
    lossSum = 0
    L=len(currentOutput)
    for i in range(L-1):
        lossSum += (currentOutput[i] - currentOutput[i+1]) * (currentOutput[i] - currentOutput[i+1])
    return lossSum.tr()/L

def makeTrainingStates(numOutGen,numTrainingData):
    #TODO idea
    trainingData = []
    for i in range(numTrainingData):
        t = qt.basis(2**numOutGen, randint(0, numOutGen-1))
        dims1 = [2 for i in range(numOutGen)]
        dims2 = [1 for i in range(numOutGen)]
        dims = [dims1, dims2]
        t.dims = dims
        trainingData.append(t)
    return trainingData

BellStates = []
BellStatesNotNormed = []
t=qt.basis(2 ** 2, 0)+qt.basis(2 ** 2, 3)
BellStatesNotNormed.append(t)
t = qt.basis(2 ** 2, 0) - qt.basis(2 ** 2, 3)
BellStatesNotNormed.append(t)
t = qt.basis(2 ** 2, 1) + qt.basis(2 ** 2, 2)
BellStatesNotNormed.append(t)
t = qt.basis(2 ** 2, 1) - qt.basis(2 ** 2, 2)
BellStatesNotNormed.append(t)
dims1 = [2 for i in range(2)]
dims2 = [1 for i in range(2)]
dims = [dims1, dims2]
for t in BellStatesNotNormed:
    s = (1 / sc.linalg.norm(t)) * t
    s.dims = dims
    BellStates.append(s)

def line(n,dim): # number of states, # dimension of state
    data = []
    lineIndex = list(range(0, n))
    for i in range(n):
        t = ((lineIndex[-i - 1]) / (n- 1)) * qt.basis(2 ** dim, 0) + ((lineIndex[i]) / (n - 1)) * qt.basis(2 ** dim, 1)
        t = (1 / sc.linalg.norm(t)) * t
        dims1 = [2 for i in range(dim)]
        dims2 = [1 for i in range(dim)]
        dims = [dims1, dims2]
        t.dims = dims
        data.append(t)
    # shuffle(data) # shuffling will be done in main function
    return data

def connectedClusters(n,dim): # number of states, # dimension of state
    dims1 = [2 for i in range(dim)]
    dims2 = [1 for i in range(dim)]
    dims = [dims1, dims2]
    lenCluster1=int((n-1)/2)
    lenCluster2=n-lenCluster1-1
    width = 2
    # data for clusters (list of n * width elements)
    data = []
    lineIndex = list(range(0, n * width))
    for i in range(n * width):  # -1 when its not a circle
        t = ((lineIndex[-i - 1]) / (n * width - 1)) * qt.basis(2 ** dim, 0) + (
                (lineIndex[i]) / (n * width - 1)) * qt.basis(2 ** dim, 1)
        t = (1 / sc.linalg.norm(t)) * t
        t.dims = dims
        data.append(t)
    # cluster 1
    cluster1 = data[:lenCluster1]
    # cluster 2
    cluster2 = data[-lenCluster2:]
    #connectionState
    connectionState =  qt.basis(2 ** dim, 0) + qt.basis(2 ** dim, 1)
    connectionState = (1 / sc.linalg.norm(connectionState)) * connectionState
    connectionState.dims=dims
    return cluster1+[connectionState]+cluster2


#### concentrableEntanglementMean part
from itertools import chain, combinations
import math

def clusters(n,dim): # number of states, # dimension of state
    dims1 = [2 for i in range(dim)]
    dims2 = [1 for i in range(dim)]
    dims = [dims1, dims2]
    lenCluster1=int((n-1)/2)
    lenCluster2=n-lenCluster1
    width = 3
    # data for clusters (list of n * width elements)
    data = []
    lineIndex = list(range(0, n * width))
    for i in range(n * width):  # -1 when its not a circle
        t = ((lineIndex[-i - 1]) / (n * width - 1)) * qt.basis(2 ** dim, 0) + (
                (lineIndex[i]) / (n * width - 1)) * qt.basis(2 ** dim, 1)
        t = (1 / sc.linalg.norm(t)) * t
        t.dims = dims
        data.append(t)
    # cluster 1
    cluster1 = data[:lenCluster1]
    # cluster 2
    cluster2 = data[-lenCluster2:]
    return cluster1+cluster2

def getPowerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

def partialTraceKeep(obj, keep):
    # return partial trace:
    res = obj;
    if len(keep) != len(obj.dims[0]):
        res = obj.ptrace(keep);
    return res;

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

def getConcentrableEntanglementState(state) -> float:
    densityMatrix=state*state.dag()
    num_qubits = len(state.dims[1])
    powerset = getPowerset(range(num_qubits))[1:]
    sum = 1  # start there because of tr(1) that is removed from powerset
    for keep in powerset:
        if len(keep) == num_qubits:
            reduced_state = densityMatrix
        else:
            reduced_state = partialTraceKeep(densityMatrix, list(keep))
        trace = (reduced_state ** 2).tr()
        sum += trace
    return 1 - 1 / (2 ** num_qubits) * sum

def getConcentrableEntanglementDensitymatrix(state) -> float:
    densityMatrix=state
    num_qubits = len(state.dims[1])
    powerset = getPowerset(range(num_qubits))[1:]
    sum = 1  # start there because of tr(1) that is removed from powerset
    for keep in powerset:
        if len(keep) == num_qubits:
            reduced_state = densityMatrix
        else:
            reduced_state = partialTraceKeep(densityMatrix, list(keep))
        trace = (reduced_state ** 2).tr()
        sum += trace
    return 1 - 1 / (2 ** num_qubits) * sum

def getConcentrableEntanglementDensitymatrixFirstSummand(state) -> float:
    densityMatrix=state
    num_qubits = len(state.dims[1])
    powerset = getPowerset(range(num_qubits))[1:]
    sum = 1  # start there because of tr(1) that is removed from powerset
    keep = powerset[0]
    if len(keep) == num_qubits:
        reduced_state = densityMatrix
    else:
        reduced_state = partialTraceKeep(densityMatrix, list(keep))
    trace = (reduced_state ** 2).tr()
    sum += trace
    return 1 - 1 / (2 ** num_qubits) * sum

def getConcentrableEntanglementStateSet(num_qubits, num_states, ce_mean, ce_variance=0.05):
    training_set = []
    for _ in range(num_states):
        state_accepted = False
        while not state_accepted:
            state = randomQubitState(num_qubits)
            ce = getConcentrableEntanglementState(state)
            val = np.exp(-((ce - ce_mean) / ce_variance) ** 2)  # normal distribution
            state_accepted = np.random.uniform() < val
        training_set.append(state)
    return training_set

def concentrableEntanglementMean(states):
    costSum = 0
    if len(states) == 0:  # new
        return 1
    for state in states:
        costSum += getConcentrableEntanglementDensitymatrix(state)
    return costSum/ len(states)

def concentrableEntanglementFirstSummandMean(states):
    costSum = 0
    if len(states) == 0:  # new
        return 1
    for state in states:
        costSum += getConcentrableEntanglementDensitymatrixFirstSummand(state)
    return costSum/ len(states)


def blochSphere(outputStatesGen, fileName,trainingRoundNr):
    density_matrices = []
    for state in outputStatesGen:
        state = np.array(state)
        if state.ndim != 2:
            state = np.outer(state, state.conjugate())
        density_matrices.append(Qobj(state))
    num_qubits = int(math.log(density_matrices[0].shape[0], 2))
    fig, axes = plt.subplots(ncols=num_qubits, figsize=(num_qubits * 5, 5), subplot_kw=dict(projection='3d'))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
        for qubit in range(num_qubits):
            qubits_density_matrices = [partialTraceKeep(dm, qubit) for dm in
                                       density_matrices] if num_qubits > 1 else density_matrices
            axes[qubit].set_box_aspect((1, 1, 1))
            b = Bloch(fig=fig, axes=axes[qubit])
            b.add_states(qubits_density_matrices)
            b.render(fig=fig, axes=axes[qubit])
        fig.suptitle('Training epoche '+str(trainingRoundNr), fontsize=20)
        plt.savefig(fileName + '_bloch.png', dpi=150)
        plt.close(fig)
    density = []
    for state in density_matrices:
        density.append(state.data)
    df = pd.DataFrame({'density_matrices': density})
    df.to_csv(fileName + '_GenOutput.csv',index=False)


def QGANStatistics(blochPossible,fileName, trainingRoundNr,trainingData, numTestingData, numTest, numTrainingData, numIn, qnnArchGen, networkUnitariesGen, testingData, indexDataTest, indexDataTrain, qnnArchGenString, qnnArchDisString):
    countOutTest = [0 for i in range(numTestingData)]
    countOutTrain = [0 for i in range(numTrainingData)]
    countOut=countOutTest+countOutTrain
    data=testingData+trainingData
    incomes = randomStates(numIn, numTest)
    outcomes = feedforwardGAN(qnnArchGen, networkUnitariesGen, incomes)
    outputStates = []
    for i in range(len(outcomes)):
        outputStates.append(outcomes[i][-1])
    with open(fileName+ '_GenOutput.txt', "wb") as fp:
        pickle.dump(outputStates, fp)
    for outputState in outputStates:
        maxFidList = []
        for state in data:
            maxFidList.append((state.dag() * outputState * state).tr())
        maxPos = maxFidList.index(max(maxFidList))
        countOut[maxPos] = countOut[maxPos] + 1
    plt.bar(indexDataTest, countOut[:numTestingData], color='green', width=0.4, label='USV')
    plt.bar(indexDataTrain, countOut[numTestingData:], color='orange', width=0.4, label='SV')
    plt.ylabel("count")
    plt.xlabel("closest data state")
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='upper right',
               title='QGAN, qnnArchGen =' + qnnArchGenString + ', qnnArchDis =' + qnnArchDisString)
    # plt.show()
    plt.suptitle('Training epoche ' + str(trainingRoundNr), fontsize=20)
    plt.savefig(fileName + '_statistics.png', dpi=150)
    plt.close()
    df = pd.DataFrame(
        {'indexDataTrain': indexDataTrain, 'countOutTrain': countOut[numTestingData:]})
    df.to_csv(fileName + '_statisticsSV.csv',
              index=False)
    df = pd.DataFrame(
        {'indexDataTest': indexDataTest, 'countOutTest': countOut[:numTestingData]})
    df.to_csv(fileName + '_statisticsUSV.csv',
              index=False)
    if blochPossible:
        blochSphere(outputStates, fileName, trainingRoundNr)

def QGAN(qnnArch, lastLayerOfGen, data, lda, ep, trainingRounds, trainingRoundsGen,
         trainingRoundsDis,numTestingData,numTest,statRounds,trainingDataName):

    # preparing
    numData=len(data)
    numTrainingData=numData-numTestingData
    if 0>numTrainingData:
        print("numTestingData>numTrainingData")
        quit()

    SVindexList= list(np.ones(numTrainingData)) + list(np.zeros(numTestingData))
    shuffle(SVindexList)

    trainingData=[]
    testingData=[]
    indexDataTrain=[] # for statistics
    indexDataTest=[] # for statistics
    for i in range(numTestingData+numTrainingData):
        state = data[i]
        if SVindexList[i]==1.0:
            trainingData.append(state)
            indexDataTrain.append(i+1)  # for statistics

        else:
            testingData.append(state)
            indexDataTest.append(i + 1)  # for statistics

    # for statistics
    trainingDataStat=trainingData.copy() # for statistics
    testingDataStat=testingData.copy() # for statistics
    shuffle(trainingData)
    shuffle(testingData)

    numIn = qnnArch[0]
    numOutGen = qnnArch[lastLayerOfGen - 1]  # number of qubits of output of generator = number of qubits input of discriminator
    qnnArchGen = qnnArch[:lastLayerOfGen]
    qnnArchDis = qnnArch[lastLayerOfGen - 1:]
    inputData = randomStates(numIn, numTrainingData)
    # for plots and savings
    qnnArchGenString = ''
    for i in qnnArchGen:
        qnnArchGenString += '-' + str(i)
    qnnArchGenString = qnnArchGenString[1:]
    qnnArchDisString = ''
    for i in qnnArchDis:
        qnnArchDisString += '-' + str(i)
    qnnArchDisString = qnnArchDisString[1:]
    blochPossible = False
    if qnnArch[lastLayerOfGen - 1] == 1:
        blochPossible=True

    # network initialization
    networkUnitaries = randomNetworkOnly(qnnArch)
    networkUnitariesGen = networkUnitaries[:lastLayerOfGen]
    networkUnitariesDis = [[]] + networkUnitaries[lastLayerOfGen:]  # add emtpy list at beginning for structure reasons

    # feedforward generator + discriminator
    storedStates = feedforwardGAN(qnnArch, networkUnitaries, inputData)
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])

    # feedforward generator
    storedStatesGen = feedforwardGAN(qnnArchGen, networkUnitariesGen, inputData)
    outputStatesGen = []
    for k in range(len(storedStatesGen)):
        outputStatesGen.append(storedStatesGen[k][-1])

    # feedforward discriminator
    storedStatesDis = feedforwardGAN(qnnArchDis, networkUnitariesDis, trainingData)
    outputStatesDis = []
    for k in range(len(storedStatesDis)):
        outputStatesDis.append(storedStatesDis[k][-1])

    #cost functions
    costGen=costFunctionGen(outputStates, qnnArch)
    costDis1 = costFunctionDis1(outputStates, qnnArch)
    costDis2 = costFunctionDis2(outputStatesDis, qnnArchDis)
    costTest = generatorTesting(numIn, numTest, qnnArchGen, networkUnitariesGen, testingData)
    costHilbert = costFunctionHilbertTest(outputStates)
    conEnt = concentrableEntanglementMean(outputStatesGen)
    conEntFirst = concentrableEntanglementFirstSummandMean(outputStatesGen)
    s=0
    plotlist = [[0],[costGen],[costDis1+costDis2], [costTest],[costHilbert],[conEnt],[conEntFirst]]


    for round in range(trainingRounds):
        trainingRoundNr = round + 1
        fileName = 'QGAN_' + str(numData) + 'data' + str(numTrainingData) + 'sv_' + str(numTest) + 'statData_' + str(
            numTest) + 'statData_' + qnnArchGenString + 'networkGen_' + qnnArchDisString + 'networkDis' + '_lda' + str(
            lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRoundNr) + '_roundsGen' + str(trainingRoundsGen) + '_roundsDis' + str(
            trainingRoundsDis) + '_' + trainingDataName
        # generator
        print("Training Round ",trainingRoundNr," of ",trainingRounds)
        for k in range(trainingRoundsGen):
            # make training data set
            trainingDataSet=[]
            for i in range(len(trainingData)):
                t = trainingData[randint(0,len(trainingData)-1)]
                trainingDataSet.append(t)
            # make random input data
            inputData = randomStates(numIn, numTrainingData)
            # feedforward generator + discriminator
            storedStates = feedforwardGAN(qnnArch, networkUnitaries, inputData)
            outputStates = []
            for k in range(len(storedStates)):
                outputStates.append(storedStates[k][-1])

            # training generator
            newNetworkUnitaries = qnnTrainingGen(qnnArch, qnnArchGen, networkUnitaries, storedStates, lda,ep,trainingData) #TODO: trained all unitaries, not efficient
            networkUnitariesGen=newNetworkUnitaries[:lastLayerOfGen]
            networkUnitaries = networkUnitariesGen + networkUnitariesDis[1:]  # remove emtpy list at beginning for structure reasons

            # feedforward generator + discriminator
            storedStates = feedforwardGAN(qnnArch, networkUnitaries, inputData)
            outputStates = []
            for k in range(len(storedStates)):
                outputStates.append(storedStates[k][-1])

            # feedforward generator
            storedStatesGen = feedforwardGAN(qnnArchGen, networkUnitariesGen, inputData)
            outputStatesGen = []
            for k in range(len(storedStatesGen)):
                outputStatesGen.append(storedStatesGen[k][-1])

            # feedforward discriminator
            storedStatesDis = feedforwardGAN(qnnArchDis, networkUnitariesDis, trainingData)
            outputStatesDis = []
            for k in range(len(storedStatesDis)):
                outputStatesDis.append(storedStatesDis[k][-1])

            # cost functions
            costGen = costFunctionGen(outputStates, qnnArch)
            costDis1 = costFunctionDis1(outputStates, qnnArch)
            costDis2 = costFunctionDis2(outputStatesDis, qnnArchDis)
            costTest = generatorTesting(numIn, numTest, qnnArchGen, networkUnitariesGen, testingData)
            costHilbert = costFunctionHilbertTest(outputStates)
            conEnt = concentrableEntanglementMean(outputStatesGen)
            conEntFirst = concentrableEntanglementFirstSummandMean(outputStatesGen)
            s=s+ep
            plotlist[0].append(s)
            plotlist[1].append(costGen)
            plotlist[2].append(costDis1+costDis2)
            plotlist[3].append(costTest)
            plotlist[4].append(costHilbert)
            plotlist[5].append(conEnt)
            plotlist[6].append(conEntFirst)

        # discriminator
        for k in range(trainingRoundsDis):
            # make random input data
            inputData = randomStates(numIn, numTrainingData)
            # training discriminator
            newNetworkUnitaries = qnnTrainingDis(qnnArch, qnnArchGen, networkUnitaries, storedStates,storedStatesDis, lda,ep,trainingData)
            networkUnitariesDis = [[]] + newNetworkUnitaries[lastLayerOfGen:]
            networkUnitaries = networkUnitariesGen + networkUnitariesDis[1:]

            # feedforward generator + discriminator
            storedStates = feedforwardGAN(qnnArch, networkUnitaries, inputData)
            outputStates = []
            for k in range(len(storedStates)):
                outputStates.append(storedStates[k][-1])

            # feedforward generator
            storedStatesGen = feedforwardGAN(qnnArchGen, networkUnitariesGen, inputData)
            outputStatesGen = []
            for k in range(len(storedStatesGen)):
                outputStatesGen.append(storedStatesGen[k][-1])

            # feedforward discriminator
            storedStatesDis = feedforwardGAN(qnnArchDis, networkUnitariesDis, trainingDataSet)
            outputStatesDis = []
            for k in range(len(storedStatesDis)):
                outputStatesDis.append(storedStatesDis[k][-1])

            # cost functions
            costGen = costFunctionGen(outputStates, qnnArch)
            costDis1 = costFunctionDis1(outputStates, qnnArch)
            costDis2 = costFunctionDis2(outputStatesDis, qnnArchDis)
            costTest = generatorTesting(numIn, numTest, qnnArchGen, networkUnitariesGen, trainingData)
            conEnt = concentrableEntanglementMean(outputStatesGen)
            conEntFirst = concentrableEntanglementFirstSummandMean(outputStatesGen)
            s=s+ep
            plotlist[0].append(s)
            plotlist[1].append(costGen)
            plotlist[2].append(costDis1+costDis2)
            plotlist[3].append(costTest)
            plotlist[4].append(costHilbert)
            plotlist[5].append(conEnt)
            plotlist[6].append(conEntFirst)
        if (trainingRoundNr)%statRounds==0:
            print("Statistics of Training Round ",trainingRoundNr)
            QGANStatistics(blochPossible,fileName, trainingRoundNr,trainingData, numTestingData, numTest, numTrainingData, numIn, qnnArchGen, networkUnitariesGen, testingData, indexDataTest, indexDataTrain, qnnArchGenString, qnnArchDisString)
    plt.scatter(plotlist[0], plotlist[1], label='costFunctionGen', color='orange', s=10)
    plt.scatter(plotlist[0], plotlist[2], label='costFunctionDis', color='green', s=10)
    plt.scatter(plotlist[0], plotlist[3], label='costFunctionTest', color='red', s=10)
    plt.scatter(plotlist[0], plotlist[4], label='costHilbert', color='black', s=10)
    plt.scatter(plotlist[0], plotlist[5], label='conEnt', color='blue', s=10)
    plt.scatter(plotlist[0], plotlist[6], label='conEntFirstSummand', color='violet', s=10)
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='upper left',
               title='QGAN, qnnArchGen =' + qnnArchGenString + ', qnnArchDis =' + qnnArchDisString)
    plt.xlabel("s")
    plt.ylabel("cost")
    #plt.show()
    plt.savefig(fileName +'_training.png',bbox_inches='tight', dpi=150)
    plt.close()
    df = pd.DataFrame(
        {'step times epsilon':plotlist[0],'costFunctionGen': plotlist[1], 'costFunctionDis': plotlist[2], 'costFunctionTest': plotlist[3], 'costHilbert': plotlist[4]})
    df.to_csv(fileName +'_training.csv',
              index=False)
    return outputStates


"""
numData=50
numTestingData=40
rounds=1000
numTest=100
statRounds=100

cluster1 = connectedClusters(numData,1)
cluster2 = connectedClusters(numData,2)
cluster3 = connectedClusters(numData,3)
QGAN([1, 1, 1], 2, cluster1, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "line")
QGAN([1, 2, 1], 2, cluster2, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "line")
QGAN([1, 3, 1], 2, cluster3, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "line")
QGAN([2, 3, 2], 2, cluster3, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "line")

line1 = line(numData,1)
line2 = line(numData,2)
line3 = line(numData,3)
QGAN([1, 1, 1], 2, line1, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "line")
QGAN([1, 2, 1], 2, line2, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "line")
QGAN([1, 3, 1], 2, line3, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "line")
QGAN([2, 3, 2], 2, line3, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "line")


conEntStates2 = getConcentrableEntanglementStateSet(2, numData, 0.25, ce_variance=0.05)
conEntStates3 = getConcentrableEntanglementStateSet(3, numData, 0.25, ce_variance=0.05)
QGAN([1, 2, 1], 2, conEntStates2, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "conEntStates0i25")
QGAN([1, 3, 1], 2, conEntStates3, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "conEntStates0i25")
conEntStates2 = getConcentrableEntanglementStateSet(2, numData, 0.5, ce_variance=0.05)
conEntStates3 = getConcentrableEntanglementStateSet(3, numData, 0.5, ce_variance=0.05)
QGAN([1, 2, 1], 2, conEntStates2, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "conEntStates0i5")
QGAN([1, 3, 1], 2, conEntStates3, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "conEntStates0i5")
conEntStates2 = getConcentrableEntanglementStateSet(2, numData, 0.1, ce_variance=0.05)
conEntStates3 = getConcentrableEntanglementStateSet(3, numData, 0.1, ce_variance=0.05)
QGAN([1, 2, 1], 2, conEntStates2, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "conEntStates0i1")
QGAN([1, 3, 1], 2, conEntStates3, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "conEntStates0i1")

"""

def QGANMeans(qnnArch, lastLayerOfGen, data, lda, ep, trainingRounds, trainingRoundsGen, trainingRoundsDis,numTestingData,numTest,statRounds,trainingDataName,meanNum):
    for i in range(1, meanNum + 1):
        meanName = trainingDataName + '_plot' + str(i)
        print(meanName)
        QGAN(qnnArch, lastLayerOfGen, data, lda, ep, trainingRounds, trainingRoundsGen, trainingRoundsDis,numTestingData, numTest, statRounds, meanName)

    numData = len(data)
    qnnArchGen = qnnArch[:lastLayerOfGen]
    qnnArchDis = qnnArch[lastLayerOfGen - 1:]
    # for plots and savings
    qnnArchGenString = ''
    for i in qnnArchGen:
        qnnArchGenString += '-' + str(i)
    qnnArchGenString = qnnArchGenString[1:]
    qnnArchDisString = ''
    for i in qnnArchDis:
        qnnArchDisString += '-' + str(i)
    qnnArchDisString = qnnArchDisString[1:]
    numTrainingData=numData-numTestingData
    for trainingRoundNr in range(1,trainingRounds+1):
        if (trainingRoundNr) % statRounds == 0:
            fileName = 'QGAN_' + str(numData) + 'data' + str(numTrainingData) + 'sv_' + str(numTest) + 'statData_' + str(
                numTest) + 'statData_' + qnnArchGenString + 'networkGen_' + qnnArchDisString + 'networkDis' + '_lda' + str(
                lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
                trainingRoundNr) + '_roundsGen' + str(trainingRoundsGen) + '_roundsDis' + str(
                trainingRoundsDis) + '_' + trainingDataName
            countTT = [0 for i in range(numData)]
            for i in range(1,meanNum+1):
                dfSV = pd.read_csv(fileName+'_plot'+str(i)+"_statisticsSV.csv")
                dfUSV = pd.read_csv(fileName+'_plot'+str(i)+"_statisticsUSV.csv")
                for row in dfSV.itertuples():
                    countTT[row.indexDataTrain-1]=countTT[row.indexDataTrain-1]+row.countOutTrain
                for row in dfUSV.itertuples():
                    countTT[row.indexDataTest-1]=countTT[row.indexDataTest-1]+row.countOutTest

            countTTMean = [0 for i in range(numData)]
            for i in range(numData):
                countTTMean[i]=countTT[i]/meanNum
            df = pd.DataFrame(
                {'index': range(1, len(countTTMean) + 1), 'countTTMean': countTTMean})
            df.to_csv(fileName + "_statMean.CSV", index=False)
            plt.bar(range(1,len(countTTMean)+1), countTTMean, color='red', width=0.4, label='SV+USV')
            plt.ylabel("count")
            plt.xlabel("closest data state")
            plt.legend(title_fontsize='x-small', fontsize='x-small', loc='upper right',
                       title='QGAN, qnnArchGen =' + qnnArchGenString + ', qnnArchDis =' + qnnArchDisString)
            #plt.show()
            plt.suptitle('Training epoche ' + str(trainingRoundNr), fontsize=20)
            plt.savefig(fileName+"_statMean.png", dpi=150)
            plt.close()

# vs line:

def QGANStatisticsVsLine(blochPossible,fileName, trainingRoundNr,trainingData, numTestingData, numTest, numTrainingData, numIn, qnnArchGen, networkUnitariesGen, testingData, indexDataTest, indexDataTrain, qnnArchGenString, qnnArchDisString):
    countOutTest = [0 for i in range(numTestingData)]
    countOutTrain = [0 for i in range(numTrainingData)]
    countOut=countOutTest+countOutTrain
    dataC=testingData+trainingData
    n=len(dataC)
    data=line(n,qnnArchGen[-1])
    incomes = randomStates(numIn, numTest)
    outcomes = feedforwardGAN(qnnArchGen, networkUnitariesGen, incomes)
    outputStates = []
    for i in range(len(outcomes)):
        outputStates.append(outcomes[i][-1])
    with open(fileName+ '_GenOutput.txt', "wb") as fp:
        pickle.dump(outputStates, fp)
    for outputState in outputStates:
        maxFidList = []
        for state in data:
            maxFidList.append((state.dag() * outputState * state).tr())
        maxPos = maxFidList.index(max(maxFidList))
        countOut[maxPos] = countOut[maxPos] + 1
    index=range(1,len(data)+1)
    plt.bar(index, countOut, color='blue', width=0.4, label='count')
    plt.ylabel("count")
    plt.xlabel("closest data state")
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='upper right',
               title='QGAN, qnnArchGen =' + qnnArchGenString + ', qnnArchDis =' + qnnArchDisString)
    # plt.show()
    plt.suptitle('Training epoche ' + str(trainingRoundNr), fontsize=20)
    plt.savefig(fileName + '_statistics.png', dpi=150)
    plt.close()
    df = pd.DataFrame(
        {'index': index, 'countOut': countOut})
    df.to_csv(fileName + '_statistics.csv',
              index=False)
    if blochPossible:
        blochSphere(outputStates, fileName, trainingRoundNr)

def QGANVsLine(qnnArch, lastLayerOfGen, data, lda, ep, trainingRounds, trainingRoundsGen,
         trainingRoundsDis,numTestingData,numTest,statRounds,trainingDataName):

    # preparing
    numData=len(data)
    numTrainingData=numData-numTestingData
    if 0>numTrainingData:
        print("numTestingData>numTrainingData")
        quit()

    SVindexList= list(np.ones(numTrainingData)) + list(np.zeros(numTestingData))
    shuffle(SVindexList)

    trainingData=[]
    testingData=[]
    indexDataTrain=[] # for statistics
    indexDataTest=[] # for statistics
    for i in range(numTestingData+numTrainingData):
        state = data[i]
        if SVindexList[i]==1.0:
            trainingData.append(state)
            indexDataTrain.append(i+1)  # for statistics

        else:
            testingData.append(state)
            indexDataTest.append(i + 1)  # for statistics

    # for statistics
    trainingDataStat=trainingData.copy() # for statistics
    testingDataStat=testingData.copy() # for statistics
    shuffle(trainingData)
    shuffle(testingData)

    numIn = qnnArch[0]
    numOutGen = qnnArch[lastLayerOfGen - 1]  # number of qubits of output of generator = number of qubits input of discriminator
    qnnArchGen = qnnArch[:lastLayerOfGen]
    qnnArchDis = qnnArch[lastLayerOfGen - 1:]
    inputData = randomStates(numIn, numTrainingData)
    # for plots and savings
    qnnArchGenString = ''
    for i in qnnArchGen:
        qnnArchGenString += '-' + str(i)
    qnnArchGenString = qnnArchGenString[1:]
    qnnArchDisString = ''
    for i in qnnArchDis:
        qnnArchDisString += '-' + str(i)
    qnnArchDisString = qnnArchDisString[1:]
    blochPossible = False
    if qnnArch[lastLayerOfGen - 1] == 1:
        blochPossible=True

    # network initialization
    networkUnitaries = randomNetworkOnly(qnnArch)
    networkUnitariesGen = networkUnitaries[:lastLayerOfGen]
    networkUnitariesDis = [[]] + networkUnitaries[lastLayerOfGen:]  # add emtpy list at beginning for structure reasons

    # feedforward generator + discriminator
    storedStates = feedforwardGAN(qnnArch, networkUnitaries, inputData)
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])

    # feedforward generator
    storedStatesGen = feedforwardGAN(qnnArchGen, networkUnitariesGen, inputData)
    outputStatesGen = []
    for k in range(len(storedStatesGen)):
        outputStatesGen.append(storedStatesGen[k][-1])

    # feedforward discriminator
    storedStatesDis = feedforwardGAN(qnnArchDis, networkUnitariesDis, trainingData)
    outputStatesDis = []
    for k in range(len(storedStatesDis)):
        outputStatesDis.append(storedStatesDis[k][-1])

    #cost functions
    costGen=costFunctionGen(outputStates, qnnArch)
    costDis1 = costFunctionDis1(outputStates, qnnArch)
    costDis2 = costFunctionDis2(outputStatesDis, qnnArchDis)
    costTest = generatorTesting(numIn, numTest, qnnArchGen, networkUnitariesGen, testingData)
    costHilbert = costFunctionHilbertTest(outputStates)
    conEnt = concentrableEntanglementMean(outputStatesGen)
    conEntFirst = concentrableEntanglementFirstSummandMean(outputStatesGen)
    s=0
    plotlist = [[0],[costGen],[costDis1+costDis2], [costTest],[costHilbert],[conEnt],[conEntFirst]]


    for round in range(trainingRounds):
        trainingRoundNr = round + 1
        fileName = 'QGAN_' + str(numData) + 'data' + str(numTrainingData) + 'sv_' + str(numTest) + 'statData_' + str(
            numTest) + 'statData_' + qnnArchGenString + 'networkGen_' + qnnArchDisString + 'networkDis' + '_lda' + str(
            lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRoundNr) + '_roundsGen' + str(trainingRoundsGen) + '_roundsDis' + str(
            trainingRoundsDis) + '_' + trainingDataName
        # generator
        print("Training Round ",trainingRoundNr," of ",trainingRounds)
        for k in range(trainingRoundsGen):
            # make training data set
            trainingDataSet=[]
            for i in range(len(trainingData)):
                t = trainingData[randint(0,len(trainingData)-1)]
                trainingDataSet.append(t)
            # make random input data
            inputData = randomStates(numIn, numTrainingData)
            # feedforward generator + discriminator
            storedStates = feedforwardGAN(qnnArch, networkUnitaries, inputData)
            outputStates = []
            for k in range(len(storedStates)):
                outputStates.append(storedStates[k][-1])

            # training generator
            newNetworkUnitaries = qnnTrainingGen(qnnArch, qnnArchGen, networkUnitaries, storedStates, lda,ep,trainingData) #TODO: trained all unitaries, not efficient
            networkUnitariesGen=newNetworkUnitaries[:lastLayerOfGen]
            networkUnitaries = networkUnitariesGen + networkUnitariesDis[1:]  # remove emtpy list at beginning for structure reasons

            # feedforward generator + discriminator
            storedStates = feedforwardGAN(qnnArch, networkUnitaries, inputData)
            outputStates = []
            for k in range(len(storedStates)):
                outputStates.append(storedStates[k][-1])

            # feedforward generator
            storedStatesGen = feedforwardGAN(qnnArchGen, networkUnitariesGen, inputData)
            outputStatesGen = []
            for k in range(len(storedStatesGen)):
                outputStatesGen.append(storedStatesGen[k][-1])

            # feedforward discriminator
            storedStatesDis = feedforwardGAN(qnnArchDis, networkUnitariesDis, trainingData)
            outputStatesDis = []
            for k in range(len(storedStatesDis)):
                outputStatesDis.append(storedStatesDis[k][-1])

            # cost functions
            costGen = costFunctionGen(outputStates, qnnArch)
            costDis1 = costFunctionDis1(outputStates, qnnArch)
            costDis2 = costFunctionDis2(outputStatesDis, qnnArchDis)
            costTest = generatorTesting(numIn, numTest, qnnArchGen, networkUnitariesGen, testingData)
            costHilbert = costFunctionHilbertTest(outputStates)
            conEnt = concentrableEntanglementMean(outputStatesGen)
            conEntFirst = concentrableEntanglementFirstSummandMean(outputStatesGen)
            s=s+ep
            plotlist[0].append(s)
            plotlist[1].append(costGen)
            plotlist[2].append(costDis1+costDis2)
            plotlist[3].append(costTest)
            plotlist[4].append(costHilbert)
            plotlist[5].append(conEnt)
            plotlist[6].append(conEntFirst)

        # discriminator
        for k in range(trainingRoundsDis):
            # make random input data
            inputData = randomStates(numIn, numTrainingData)
            # training discriminator
            newNetworkUnitaries = qnnTrainingDis(qnnArch, qnnArchGen, networkUnitaries, storedStates,storedStatesDis, lda,ep,trainingData)
            networkUnitariesDis = [[]] + newNetworkUnitaries[lastLayerOfGen:]
            networkUnitaries = networkUnitariesGen + networkUnitariesDis[1:]

            # feedforward generator + discriminator
            storedStates = feedforwardGAN(qnnArch, networkUnitaries, inputData)
            outputStates = []
            for k in range(len(storedStates)):
                outputStates.append(storedStates[k][-1])

            # feedforward generator
            storedStatesGen = feedforwardGAN(qnnArchGen, networkUnitariesGen, inputData)
            outputStatesGen = []
            for k in range(len(storedStatesGen)):
                outputStatesGen.append(storedStatesGen[k][-1])

            # feedforward discriminator
            storedStatesDis = feedforwardGAN(qnnArchDis, networkUnitariesDis, trainingDataSet)
            outputStatesDis = []
            for k in range(len(storedStatesDis)):
                outputStatesDis.append(storedStatesDis[k][-1])

            # cost functions
            costGen = costFunctionGen(outputStates, qnnArch)
            costDis1 = costFunctionDis1(outputStates, qnnArch)
            costDis2 = costFunctionDis2(outputStatesDis, qnnArchDis)
            costTest = generatorTesting(numIn, numTest, qnnArchGen, networkUnitariesGen, trainingData)
            conEnt = concentrableEntanglementMean(outputStatesGen)
            conEntFirst = concentrableEntanglementFirstSummandMean(outputStatesGen)
            s=s+ep
            plotlist[0].append(s)
            plotlist[1].append(costGen)
            plotlist[2].append(costDis1+costDis2)
            plotlist[3].append(costTest)
            plotlist[4].append(costHilbert)
            plotlist[5].append(conEnt)
            plotlist[6].append(conEntFirst)
        if (trainingRoundNr)%statRounds==0:
            print("Statistics of Training Round ",trainingRoundNr)
            QGANStatisticsVsLine(blochPossible,fileName, trainingRoundNr,trainingData, numTestingData, numTest, numTrainingData, numIn, qnnArchGen, networkUnitariesGen, testingData, indexDataTest, indexDataTrain, qnnArchGenString, qnnArchDisString)
    plt.scatter(plotlist[0], plotlist[1], label='costFunctionGen', color='orange', s=10)
    plt.scatter(plotlist[0], plotlist[2], label='costFunctionDis', color='green', s=10)
    plt.scatter(plotlist[0], plotlist[3], label='costFunctionTest', color='red', s=10)
    plt.scatter(plotlist[0], plotlist[4], label='costHilbert', color='black', s=10)
    plt.scatter(plotlist[0], plotlist[5], label='conEnt', color='blue', s=10)
    plt.scatter(plotlist[0], plotlist[6], label='conEntFirstSummand', color='violet', s=10)
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='upper left',
               title='QGAN, qnnArchGen =' + qnnArchGenString + ', qnnArchDis =' + qnnArchDisString)
    plt.xlabel("s")
    plt.ylabel("cost")
    #plt.show()
    plt.savefig(fileName +'_training.png',bbox_inches='tight', dpi=150)
    plt.close()
    df = pd.DataFrame(
        {'step times epsilon':plotlist[0],'costFunctionGen': plotlist[1], 'costFunctionDis': plotlist[2], 'costFunctionTest': plotlist[3], 'costHilbert': plotlist[4]})
    df.to_csv(fileName +'_training.csv',
              index=False)
    return outputStates

def QGANMeansVsLine(qnnArch, lastLayerOfGen, data, lda, ep, trainingRounds, trainingRoundsGen, trainingRoundsDis,numTestingData,numTest,statRounds,trainingDataName,meanNum):
    for i in range(9, meanNum + 1):
        meanName = trainingDataName + '_plot' + str(i)
        print(meanName)
        QGANVsLine(qnnArch, lastLayerOfGen, data, lda, ep, trainingRounds, trainingRoundsGen, trainingRoundsDis,numTestingData, numTest, statRounds, meanName)

    numData = len(data)
    qnnArchGen = qnnArch[:lastLayerOfGen]
    qnnArchDis = qnnArch[lastLayerOfGen - 1:]
    # for plots and savings
    qnnArchGenString = ''
    for i in qnnArchGen:
        qnnArchGenString += '-' + str(i)
    qnnArchGenString = qnnArchGenString[1:]
    qnnArchDisString = ''
    for i in qnnArchDis:
        qnnArchDisString += '-' + str(i)
    qnnArchDisString = qnnArchDisString[1:]
    numTrainingData=numData-numTestingData
    for trainingRoundNr in range(1,trainingRounds+1):
        if (trainingRoundNr) % statRounds == 0:
            fileName = 'QGAN_' + str(numData) + 'data' + str(numTrainingData) + 'sv_' + str(numTest) + 'statData_' + str(
                numTest) + 'statData_' + qnnArchGenString + 'networkGen_' + qnnArchDisString + 'networkDis' + '_lda' + str(
                lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
                trainingRoundNr) + '_roundsGen' + str(trainingRoundsGen) + '_roundsDis' + str(
                trainingRoundsDis) + '_' + trainingDataName
            countTT = [0 for i in range(numData)]
            for i in range(1,meanNum+1):
                df = pd.read_csv(fileName+'_plot'+str(i)+"_statistics.csv")
                for row in df.itertuples():
                    countTT[row.index-1]=countTT[row.index-1]+row.countOut

            countTTMean = [0 for i in range(numData)]
            for i in range(numData):
                countTTMean[i]=countTT[i]/meanNum
            df = pd.DataFrame(
                {'index': range(1, len(countTTMean) + 1), 'countTTMean': countTTMean})
            df.to_csv(fileName + "_statMean.CSV", index=False)
            plt.bar(range(1,len(countTTMean)+1), countTTMean, color='red', width=0.4, label='SV+USV')
            plt.ylabel("count")
            plt.xlabel("closest data state")
            plt.legend(title_fontsize='x-small', fontsize='x-small', loc='upper right',
                       title='QGAN, qnnArchGen =' + qnnArchGenString + ', qnnArchDis =' + qnnArchDisString)
            #plt.show()
            plt.suptitle('Training epoche ' + str(trainingRoundNr), fontsize=20)
            plt.savefig(fileName+"_statMean.png", dpi=150)
            plt.close()

numData=50
numTestingData=40
rounds=200
numTest=100
statRounds=100
meanNum=10

cluster1 = clusters(numData,1)
QGANMeansVsLine([1, 1, 1], 2, cluster1, 1, 0.01, rounds, 1, 1, numTestingData, numTest, statRounds, "CvsLi",meanNum)

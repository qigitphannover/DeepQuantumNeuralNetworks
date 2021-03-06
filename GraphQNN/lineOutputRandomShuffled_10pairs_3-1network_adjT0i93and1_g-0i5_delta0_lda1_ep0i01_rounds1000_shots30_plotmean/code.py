######  Code for Feedforward Quantum Neural Networks
# This approach uses a graph with a vertex for every training pair (produced with quantum_graph_benchmark.py).
# If the output of one training pair is close to the output of another training pairs output,
# this is represented by an edge in the graph. "Close" is here defined with a threshold and the
# hilbert-schmidt-norm.

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


### Semisupervised training with Graph


# Graph part of the training loss
def costFunctionGraph(adjMatrix, currentOutput):
    lossSum = 0
    for i in range(len(adjMatrix[0])):
        for j in range(i, len(adjMatrix[0])):
            if adjMatrix[i][j] != 0:
                lossSum += adjMatrix[i][j] * 2 * (
                        (currentOutput[i] - currentOutput[j]) * (currentOutput[i] - currentOutput[j]))
    return lossSum.tr()


# add the two update matrecies to the update matrix for semisupervised + graph training
def addUpdateMatrix(qnnArch, currentUnitaries, trainingDataSv, lda, ep, gamma, outputStates,
                    adjMatrix, storedStates, storedStatesSv, l, j):
    res = makeUpdateMatrixSv(qnnArch, currentUnitaries, trainingDataSv, storedStatesSv, lda,
                             ep, l, j) \
          + gamma * makeUpdateMatrixGraph(qnnArch, currentUnitaries, lda, outputStates,
                                          adjMatrix, storedStates, l, j)
    return ((0 + 1j) * ep * res).expm()


### Helperfunctions with graph


# make the supervised part of the update matrix
def makeUpdateMatrixSv(qnnArch, unitaries, trainingData, storedStates, lda, ep, l, j):
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
    # here is the difference to makeUpdateMatrix
    summ = ((0 + 1j) * 2 ** numInputQubits) / (lda * len(trainingData)) * summ
    return summ


# brings the update matrix in the right form to apply on the unitaries
def makeUpdateMatrixTensoredGraph(updateMatrix, qnnArch, l, m):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    if numOutputQubits - 1 != 0:
        updateMatrix = qt.tensor(updateMatrix, tensoredId(numOutputQubits - 1))

    # Return
    return swappedOp(updateMatrix, numInputQubits, numInputQubits + m)


# calculates the first part of the M^l_m graph update matrix
def updateMatrixFirstPartGraph(qnnArch, currentUnitaries, inputState, l, m):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Tensor input state
    state = qt.tensor(inputState, tensoredQubit0(numOutputQubits))

    # Calculate needed product unitary
    productUni = currentUnitaries[l][0].copy()
    for x in range(1, m + 1):
        productUni = currentUnitaries[l][x] * productUni

    # Multiply
    return productUni * state * productUni.dag()


# calculates the second part of the M^l_m graph update matrix
def updateMatrixSecondPartGraph(qnnArch, currentUnitaries, inputState, l, m):
    numInputQubits = qnnArch[l - 1]
    numOutputQubits = qnnArch[l]

    # Calculate sigma state
    state = inputState
    for y in range(len(qnnArch) - 1, l, -1):
        state = makeAdjointLayerChannel(qnnArch, currentUnitaries, y, state)

    # Tensor sigma state
    state = qt.tensor(tensoredId(numInputQubits), state)

    # Calculate needed product unitary
    productUni = tensoredId(numInputQubits + numOutputQubits)
    for x in range(m + 1, numOutputQubits):
        productUni = currentUnitaries[l][x] * productUni

    # Multiply
    return productUni.dag() * state * productUni


# make the graph part of the update matrix
def makeUpdateMatrixGraph(qnnArch, currentUnitaries, lda, currentOutput, adjMatrix,
                          storedStates, l, m):
    numInputQubits = qnnArch[l - 1]

    # Calculate the sum:
    summ = 0
    for i in range(len(adjMatrix[0])):
        for j in range(i, len(adjMatrix[0])):
            if adjMatrix[i][j] != 0:
                # Calculate the commutator
                firstPart = updateMatrixFirstPartGraph(qnnArch, currentUnitaries,
                                                       storedStates[i][l - 1] - storedStates[j][l - 1], l, m)
                secondPart = updateMatrixSecondPartGraph(qnnArch, currentUnitaries, currentOutput[i] - currentOutput[j],
                                                         l,
                                                         m)
                mat = qt.commutator(firstPart, secondPart)

                # Trace out the rest
                keep = list(range(numInputQubits))
                keep.append(numInputQubits + m)
                mat = partialTraceKeep(mat, keep)

                # Add to sum
                summ = summ + (adjMatrix[i][j] * mat)

    # Calculate the update matrix from the sum
    summ = ((0 + 1j) * (2 ** (numInputQubits + 1)) / lda) * summ
    return summ

    ### Semisupervised training with Graph


# semisupervised QNN training with Graph, outputs lists of the loss functions and the network unitaries
def qnnTrainingGraph(qnnArch, initialUnitaries, trainingData, trainingDataSv, listSv, trainingDataUsv, listUsv, lda, ep,
                     trainingRounds, adjMatrix, gamma, adjThreshold, adjThreshold2, alert=0):
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

    # Cost calculation for given unitaries for the first time
    outputStates = []
    for k in range(len(storedStates)):
        outputStates.append(storedStates[k][-1])

    outputStatesSv = []
    for k in range(len(storedStatesSv)):
        outputStatesSv.append(storedStatesSv[k][-1])

    outputStatesUsv = []
    for k in range(len(storedStatesUsv)):
        outputStatesUsv.append(storedStatesUsv[k][-1])

    # save cost values for the first time

    costValueSv = costFunction(trainingDataSv, outputStatesSv)
    testingCostAll = costFunction(trainingData, outputStates)
    # testingCostUsv = costFunction(trainingDataUsv, outputStatesUsv)
    testingCostUsv = 1
    if (N - S) != 0:
        testingCostUsv = N / (N - S) * (testingCostAll - S / (N) * costValueSv)
    costValueGraph = gamma * costFunctionGraph(adjMatrix, outputStates)
    trainingCost = costValueSv + costValueGraph

    # print(outputStates[0])
    # print(outputStates[0].dag())
    # print(trainingData[0][1])
    # print(trainingData[0][1].dag())

    plotlistGraph = [[s], [trainingCost], [testingCostAll], [testingCostUsv]]

    # Optional
    runtime = time()

    # Training of the Quantum Neural Network
    for k in range(trainingRounds):
        if alert > 0 and k % alert == 0: print("In training round " + str(k))

        ### UPDATING
        newUnitaries = unitariesCopy(currentUnitaries)

        # Loop over layers:
        for l in range(1, len(qnnArch)):
            numOutputQubits = qnnArch[l]

            # Loop over perceptrons
            for j in range(numOutputQubits):
                updateMatrix = addUpdateMatrix(qnnArch, currentUnitaries, trainingDataSv, lda, ep, gamma, outputStates,
                                               adjMatrix, storedStates, storedStatesSv, l, j)
                newUnitaries[l][j] = makeUpdateMatrixTensoredGraph(updateMatrix, qnnArch, l, j) * currentUnitaries[l][j]

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

        # Cost calculation for given unitaries in for-loop
        outputStates = []
        for k in range(len(storedStates)):
            outputStates.append(storedStates[k][-1])

        outputStatesSv = []
        for k in range(len(storedStatesSv)):
            outputStatesSv.append(storedStatesSv[k][-1])

        outputStatesUsv = []
        for k in range(len(storedStatesUsv)):
            outputStatesUsv.append(storedStatesUsv[k][-1])

        # save cost values in for-loop
        costValueSv = costFunction(trainingDataSv, outputStatesSv)
        testingCostAll = costFunction(trainingData, outputStates)
        # testingCostUsv = costFunction(trainingDataUsv, outputStatesUsv)
        testingCostUsv = 1
        if (N - S) != 0:
            testingCostUsv = N / (N - S) * (testingCostAll - S / (N) * costValueSv)
        costValueGraph = gamma * costFunctionGraph(adjMatrix, outputStates)
        trainingCost = costValueSv + costValueGraph

        plotlistGraph[0].append(s)
        plotlistGraph[1].append(trainingCost)
        plotlistGraph[2].append(testingCostAll)
        plotlistGraph[3].append(testingCostUsv)

    # Optional
    runtime = time() - runtime
    print("Trained semisupervised with graph " + str(trainingRounds) + " rounds for a " + str(
        qnnArch) + " network and " + str(
        len(trainingData)) + " training pairs in " + str(
        round(runtime, 2)) + " seconds \n with Graph related gamma = " + str(gamma) +
          " and adjThreshold = " + str(adjThreshold))

    # Return
    return [plotlistGraph, currentUnitaries]


# semisupervised QNN without AND with Graph, outputs plots and CSV
def mainSsvGraph(qnnArch, numTrainingPairs, numberSupervisedPairs, lda, ep, trainingRounds, gamma,
                 adjThreshold, adjThreshold2, delta, kind, alert=0):
    qnnArchString = ''
    for i in qnnArch:
        # append strings to the variable
        qnnArchString += '-' + str(i)
    qnnArchString = qnnArchString[1:]
    if not os.path.exists(kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                        'i') + 'and' + str(
        adjThreshold2).replace('.', 'i') + '_graph' + '.txt'):
        print('File does not exist.')
        return None
    # the number of supervised pairs has to be smaller or equal to the total number of pairs
    if numTrainingPairs < numberSupervisedPairs:
        sys.exit("Error: numTrainingPairs < numberSupervisedPairs")

    # creates network and initial unitaries
    network = randomNetwork(qnnArch, numTrainingPairs)  # TODO, not efficient
    initialUnitaries = network[1]

    # reads file produced with quantum_graph_benchmark.py
    with open(kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                        'i') + 'and' + str(
        adjThreshold2).replace('.', 'i') + '_graph' + '.txt', "rb") as fp:  # Unpickling
        graphList = pickle.load(fp)  # =[trainingData, adjMatrix, listSv, trainingDataSv, listUsv, trainingDataUsv]
    print('File ' + kind + '_' + str(numTrainingPairs) + 'pairs' + str(
        numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                    'i') + 'and' + str(
        adjThreshold2).replace('.', 'i') + '_graph' + '.txt with gamma=' + str(gamma))
    # with graphList=[trainingData, adjMatrix, listSv, trainingDataSv, listUsv, trainingDataUsv,  # save the data
    trainingData = graphList[0]
    # (trainingData)
    adjMatrix = graphList[1]
    listSv = graphList[2]
    trainingDataSv = graphList[3]
    listUsv = graphList[4]
    trainingDataUsv = graphList[5]
    fidMatrix = graphList[6]
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

    # train ssv with graph structure
    plotlistGraph = \
        qnnTrainingGraph(qnnArch, initialUnitaries, trainingData, trainingDataSv, listSv, trainingDataUsv, listUsv, lda,
                         ep,
                         trainingRounds, adjMatrix, gamma, adjThreshold, alert)[0]

    # train ssv without graph structure
    plotlistSsv = \
        qnnTrainingSsv(qnnArch, initialUnitaries, trainingData, trainingDataSv, listSv, trainingDataUsv, listUsv, lda,
                       ep,
                       trainingRounds, alert)[0]

    # plot
    for i in range(len(plotlistGraph[1])):
        if plotlistGraph[1][i] >= 0.95:
            print("Semisupervised with graph structure: Exceeds cost of 0.95 at training step " + str(i))
            break
    # training with graph can not be included in plot because negative:
    # plt.plot(plotlistGraph[0], plotlistGraph[1], label='QNN Ssv+Graph (training)', color = 'yellow')
    plt.plot(plotlistGraph[0], plotlistGraph[1], label='QNN Ssv+Graph (training)', color='orange')
    plt.plot(plotlistGraph[0], plotlistGraph[3], label='QNN Ssv+Graph (testing USV)', color='red')
    for i in range(len(plotlistSsv[1])):
        if plotlistSsv[1][i] >= 0.95:
            print("Semisupervised: Exceeds cost of 0.95 at training step " + str(i))
            break

    plt.plot(plotlistSsv[0], plotlistSsv[1], label='QNN Ssv (training)', color='green')
    plt.plot(plotlistSsv[0], plotlistSsv[3], label='QNN Ssv (testing USV)', color='blue')
    plt.xlabel("s * epsilon")
    plt.ylabel("Cost[s]")
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='lower right',
               title=qnnArchString + ', gamma =' + str(
                   gamma) + ', delta =' + str(delta))
    # plt.show()
    df = pd.DataFrame({'step times epsilon': plotlistSsv[0], 'SsvTraining': plotlistSsv[1],
                       'SsvTestingAll': plotlistSsv[2], 'SsvTestingUsv': plotlistSsv[3],
                       'SsvGraphTraining': plotlistGraph[1], 'SsvGraphTestingAll': plotlistGraph[2],
                       'SsvGraphTestingUsv': plotlistGraph[3]})

    # saves plot as figure and csv
    plt.savefig(
        kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                        'i') + 'and' + str(
            adjThreshold2).replace('.', 'i') + '_g' + str(
            gamma).replace('.', 'i') + '_delta' + str(delta).replace('.', 'i') + '_lda' + str(lda).replace('.',
                                                                                                           'i') + '_ep' + str(
            ep).replace('.', 'i') + '_plot.png', bbox_inches='tight', dpi=150)
    plt.clf()
    df.to_csv(
        kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                        'i') + 'and' + str(
            adjThreshold2).replace('.', 'i') + '_g' + str(
            gamma).replace('.', 'i') + '_delta' + str(delta).replace('.', 'i') + '_lda' + str(lda).replace('.',
                                                                                                           'i') + '_ep' + str(
            ep).replace('.', 'i') + '_plot.csv', index=False)


### Fidelity matricies for different graphs


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


### main graph functions

# plots and saves graph for every element in supervisedPairsList (different number of supervised pairs)
def graphListSv(qnnArch, numTrainingPairs, supervisedPairsList, adjThreshold, adjThreshold2, kind):
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
        print('File does not exist2.')
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

    # make adjencency matrix using the threshold and the hilbert-schmidt-norm
    adjMatrix = np.zeros([numTrainingPairs, numTrainingPairs])
    for i in range(0, numTrainingPairs):
        adjMatrix[i][i] = 1
        for j in range(0, i):
            p = fidMatrix[i][j]
            if p >= adjThreshold2:
                # oder
                # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
                # fidMatrix[i][j] = p
                # if p.tr() < adjThreshold:
                adjMatrix[i][j] = 0.5
                adjMatrix[j][i] = 0.5
            if p >= adjThreshold:
                # oder
                # p = (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()) * (trainingData[i][0]*trainingData[i][0].dag() - trainingData[j][0]*trainingData[j][0].dag()).dag()
                # fidMatrix[i][j] = p
                # if p.tr() < adjThreshold:
                adjMatrix[i][j] = 1
                adjMatrix[j][i] = 1
    summ = [sum(e) for e in adjMatrix]
    rows, cols = np.where(adjMatrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    all_rows = range(0, adjMatrix.shape[0])
    nodecolors = []
    nodelabels = {}
    nodesizes = []

    for n in listSv:
        gr.add_node(n)
        nodecolors.append('green')
        nodesizes.append(60)
        nodelabels.update({n: "sv"})
        # nodelabels.append('supervised nodes')
    for n in listUsv:
        gr.add_node(n)
        nodesizes.append(25)
        nodecolors.append('black')
        nodelabels.update({n: "usv"})
        # nodelabels.append('unsupervised nodes')
    gr.add_edges_from(edges, color='grey')
    rows2, cols2 = np.where(adjMatrix == 0.5)
    edges2 = zip(rows2.tolist(), cols2.tolist())
    gr.add_edges_from(edges2, color='lightgrey')
    pos = nx.spring_layout(gr)
    # nx.draw_networkx_nodes(gr, pos=pos, node_size=50, nodelist=listSv, node_color='green', label='supervised nodes')
    # nx.draw_networkx_nodes(gr, pos=pos, node_size=25, nodelist=listUsv, node_color='black', label='unsupervised nodes')
    edges = gr.edges()
    edgecolors = [gr[u][v]['color'] for u, v in edges]
    nodes = gr.nodes()
    # nodecolors = [gr[u]['color'] for u in nodes]
    nx.draw_networkx(gr, pos, edges=edges, edge_color=edgecolors, nodes=nodes, node_color=nodecolors,
                     node_size=nodesizes, labels=nodelabels, with_labels=False)  # 'supervised',)#nodelabels)
    # nx.draw_networkx_edges(gr, pos=pos) #, edge_color='grey')
    '''
    rows2, cols2 = np.where(adjMatrix == 0.5)
    edges2 = zip(rows2.tolist(), cols2.tolist())
    gr.add_edges_from(edges2)
    nx.draw_networkx_edges(gr2, pos=pos, edge_color='red')
    '''
    qnnArchString = ''
    for i in qnnArch:
        # append strings to the variable
        qnnArchString += '-' + str(i)
    qnnArchString = qnnArchString[1:]
    # plt.legend(title_fontsize='x-small',fontsize='x-small',loc='lower right', title=str(numTrainingPairs)+' '+str(numQubits)+'-qubit-states, threshold ='+str(adjThreshold),scatterpoints=1)
    plt.savefig(kind + '_' + str(numTrainingPairs) + 'pairs' + str(
        numSupervisedPairs) + 'sv_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                 'i') + 'and' + str(
        adjThreshold2).replace('.', 'i') + '_graph' + '.png', bbox_inches='tight', dpi=150)
    # plt.show()
    plt.clf()
    # save the plot the training data, matrix, vertex lists in a .txt using pickle
    graphList = [trainingData, adjMatrix, listSv, trainingDataSv, listUsv,
                 trainingDataUsv, fidMatrix]  # save the data
    with open(kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numSupervisedPairs) + 'sv_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                     'i') + 'and' + str(
        adjThreshold2).replace('.', 'i') + '_graph' + '.txt', "wb") as fp:  # Pickling
        pickle.dump(graphList, fp)
    # save the adjencency matrix in a latex user friedly way
    mat = str(adjMatrix)
    mat = mat.replace('[[', 'begin{pmatrix}')
    mat = mat.replace('.]]', '\end{pmatrix}')
    mat = mat.replace(']', '')
    mat = mat.replace('[', ' ')
    mat = mat.replace('.\n', '\\\\ \n')
    mat = mat.replace('.', '&')
    text_file = open(
        kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numSupervisedPairs) + 'sv_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                     'i') + 'and' + str(
            adjThreshold2).replace('.', 'i') + '_matrix' + '.txt',
        "w")
    text_file.write(mat)
    text_file.close()

# plots and saves the losses with number of supervised pairs on the x-axis
def makeLossListIndex(qnnArch, numTrainingPairs, numberSupervisedPairsList, lda, ep, trainingRounds, gamma,
                      adjThreshold, adjThreshold2, delta, index, kind):
    # load dataframe from csv
    SsvTraining = []
    SsvTestingAll = []
    SsvTestingUsv = []
    SsvGraphTraining = []
    SsvGraphTestingAll = []
    SsvGraphTestingUsv = []
    qnnArchString = ''
    for i in qnnArch:
        # append strings to the variable
        qnnArchString += '-' + str(i)
    qnnArchString = qnnArchString[1:]
    for numberSupervisedPairs in numberSupervisedPairsList:
        readdf = pd.read_csv(kind + '_' + str(numTrainingPairs) + 'pairs' + str(
            numberSupervisedPairs) + 'sv_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                        'i') + 'and' + str(
            adjThreshold2).replace('.', 'i') + '_g' + str(gamma).replace('.', 'i') + '_delta' + str(delta).replace('.',
                                                                                                                   'i') + '_lda' + str(
            lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_plot.csv')
        SsvTraining.append(float(readdf.tail(1)['SsvTraining']))
        SsvTestingAll.append(float(readdf.tail(1)['SsvTestingAll']))
        SsvTestingUsv.append(float(readdf.tail(1)['SsvTestingUsv']))
        SsvGraphTraining.append(float(readdf.tail(1)['SsvGraphTraining']))
        SsvGraphTestingAll.append(float(readdf.tail(1)['SsvGraphTestingAll']))
        SsvGraphTestingUsv.append(float(readdf.tail(1)['SsvGraphTestingUsv']))

    plt.scatter(numberSupervisedPairsList, SsvGraphTraining, label='QNN Ssv+Graph (training)', color='orange')
    plt.scatter(numberSupervisedPairsList, SsvGraphTestingUsv, label='QNN Ssv+Graph (testing USV)', color='red')
    plt.scatter(numberSupervisedPairsList, SsvTraining, label='QNN Ssv (training)', color='green')
    plt.scatter(numberSupervisedPairsList, SsvTestingUsv, label='QNN Ssv (testing USV)', color='blue')
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='lower right',
               title=qnnArchString + ', gamma =' + str(
                   gamma) + ', delta =' + str(delta))
    plt.savefig(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                                   'i') + 'and' + str(
            adjThreshold2).replace('.', 'i') + '_g' + str(gamma).replace('.', 'i') + '_delta' + str(delta).replace(
            '.', 'i') + '_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_plot' + str(
            index) + '.png', bbox_inches='tight', dpi=150)
    plt.clf()
    df = pd.DataFrame(
        {'numberSupervisedPairsList': numberSupervisedPairsList, 'SsvTraining': SsvTraining,
         'SsvTestingAll': SsvTestingAll, 'SsvTestingUsv': SsvTestingUsv, 'SsvGraphTraining': SsvGraphTraining,
         'SsvGraphTestingAll': SsvGraphTestingAll, 'SsvGraphTestingUsv': SsvGraphTestingUsv})
    df.to_csv(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                                   'i') + 'and' + str(
            adjThreshold2).replace('.', 'i') + '_g' + str(gamma).replace('.', 'i') + '_delta' + str(delta).replace(
            '.', 'i') + '_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_plot' + str(
            index) + '.csv', index=False)

    ### Fildelity matricies for different kind of graphs


# main function for QNN with graphs, plots graphs, training loss and testing loss and makes CSV file
# plots and saves the losses with number of supervised pairs on the x-axis averaged for number of shots
def makeLossListMean(qnnArch, numTrainingPairs, maxNumberSupervisedPairs, lda, ep, trainingRounds, gamma,
                     adjThreshold, adjThreshold2, delta, shots, kind):
    numberSupervisedPairsList = range(1, maxNumberSupervisedPairs + 1)
    for shotindex in range(1, shots + 1):
        # prints number of shot
        print('------ shot ' + str(shotindex) + ' of ' + str(shots) + ' ------')
        # finds the right filelity matrix function for the kind of graph and checks if enough orthogonal states
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
        numberSupervisedPairsListGraph = numberSupervisedPairsList
        for i in numberSupervisedPairsList:
            graphListSv(qnnArch, numTrainingPairs, numberSupervisedPairsListGraph, adjThreshold, adjThreshold2, kind)
            numberSupervisedPairsListGraph = numberSupervisedPairsListGraph[:-1]

        for numberSupervisedPairs in numberSupervisedPairsList:
            mainSsvGraph(qnnArch, numTrainingPairs, numberSupervisedPairs, lda, ep, trainingRounds, gamma, adjThreshold,
                         adjThreshold2, delta, kind)

        makeLossListIndex(qnnArch, numTrainingPairs, numberSupervisedPairsList, lda, ep, trainingRounds, gamma,
                          adjThreshold, adjThreshold2, delta, shotindex, kind)
    # load dataframe from csv
    SsvTrainingMeanList = []
    SsvTestingAllMeanList = []
    SsvTestingUsvMeanList = []
    SsvGraphTrainingMeanList = []
    SsvGraphTestingAllMeanList = []
    SsvGraphTestingUsvMeanList = []
    for indexSv in numberSupervisedPairsList:
        SsvTraining = []
        SsvTestingAll = []
        SsvTestingUsv = []
        SsvGraphTraining = []
        SsvGraphTestingAll = []
        SsvGraphTestingUsv = []
        qnnArchString = ''
        for i in qnnArch:
            # append strings to the variable
            qnnArchString += '-' + str(i)
        qnnArchString = qnnArchString[1:]
        for index in range(1, shots + 1):
            readdf = pd.read_csv(
                kind + '_' + str(numTrainingPairs) + 'pairs_' + qnnArchString + 'network_adjT' + str(
                    adjThreshold).replace('.',
                                          'i') + 'and' + str(
                    adjThreshold2).replace('.', 'i') + '_g' + str(gamma).replace('.', 'i') + '_delta' + str(
                    delta).replace(
                    '.', 'i') + '_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.',
                                                                                              'i') + '_rounds' + str(
                    trainingRounds) + '_plot' + str(
                    index) + '.csv')
            SsvTraining.append(readdf._get_value(indexSv - 1, 'SsvTraining'))
            SsvTestingAll.append(readdf._get_value(indexSv - 1, 'SsvTestingAll'))
            SsvTestingUsv.append(readdf._get_value(indexSv - 1, 'SsvTestingUsv'))
            SsvGraphTraining.append(readdf._get_value(indexSv - 1, 'SsvGraphTraining'))
            SsvGraphTestingAll.append(readdf._get_value(indexSv - 1, 'SsvGraphTestingAll'))
            SsvGraphTestingUsv.append(readdf._get_value(indexSv - 1, 'SsvGraphTestingUsv'))
        # means
        SsvTrainingMean = sum(SsvTraining) / len(SsvTraining)
        SsvTestingAllMean = sum(SsvTestingAll) / len(SsvTestingAll)
        SsvTestingUsvMean = sum(SsvTestingUsv) / len(SsvTestingUsv)
        SsvGraphTrainingMean = sum(SsvGraphTraining) / len(SsvGraphTraining)
        SsvGraphTestingAllMean = sum(SsvGraphTestingAll) / len(SsvGraphTestingAll)
        SsvGraphTestingUsvMean = sum(SsvGraphTestingUsv) / len(SsvGraphTestingUsv)
        # collect means in list
        SsvTrainingMeanList.append(SsvTrainingMean)
        SsvTestingAllMeanList.append(SsvTestingAllMean)
        SsvTestingUsvMeanList.append(SsvTestingUsvMean)
        SsvGraphTrainingMeanList.append(SsvGraphTrainingMean)
        SsvGraphTestingAllMeanList.append(SsvGraphTestingAllMean)
        SsvGraphTestingUsvMeanList.append(SsvGraphTestingUsvMean)

    # plots
    # or add plot SsvTrainingMean
    plt.scatter(numberSupervisedPairsList, SsvTrainingMeanList, label='QNN Ssv (training)', color='green')
    plt.scatter(numberSupervisedPairsList, SsvGraphTrainingMeanList, label='QNN Ssv+Graph (training)',
                color='orange')
    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='lower right',
               title=qnnArchString + ', gamma =' + str(
                   gamma) + ', delta =' + str(delta))
    plt.savefig(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                                   'i') + 'and' + str(
            adjThreshold2).replace('.', 'i') + '_g' + str(gamma).replace('.', 'i') + '_delta' + str(delta).replace(
            '.', 'i') + '_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_shots' + str(shots) + '_plotmean_training.png',
        bbox_inches='tight', dpi=150)
    plt.clf()
    plt.scatter(numberSupervisedPairsList, SsvTestingUsvMeanList, label='QNN Ssv (testing USV)', color='blue')
    # or add plot SsvGraphTrainingMean
    plt.scatter(numberSupervisedPairsList, SsvGraphTestingUsvMeanList, label='QNN Ssv+Graph (testing USV)', color='red')

    plt.legend(title_fontsize='x-small', fontsize='x-small', loc='lower right',
               title=qnnArchString + ', gamma =' + str(
                   gamma) + ', delta =' + str(delta))
    plt.savefig(
        kind + '_' + str(numTrainingPairs) + 'pairs_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                                   'i') + 'and' + str(
            adjThreshold2).replace('.', 'i') + '_g' + str(gamma).replace('.', 'i') + '_delta' + str(
            delta).replace(
            '.', 'i') + '_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
            trainingRounds) + '_shots' + str(shots) + '_plotmean_testing.png',
        bbox_inches='tight', dpi=150)
    plt.clf()
    df = pd.DataFrame(
        {'numberSupervisedPairsList': numberSupervisedPairsList, 'SsvTrainingMeanList': SsvTrainingMeanList,
         'SsvTestingAllMeanList': SsvTestingAllMeanList, 'SsvTestingUsvMeanList': SsvTestingUsvMeanList,
         'SsvGraphTrainingMeanList': SsvGraphTrainingMeanList,
         'SsvGraphTestingAllMeanList': SsvGraphTestingAllMeanList,
         'SsvGraphTestingUsvMeanList': SsvGraphTestingUsvMeanList})
    df.to_csv(kind + '_' +
              str(numTrainingPairs) + 'pairs_' + qnnArchString + 'network_adjT' + str(adjThreshold).replace('.',
                                                                                                            'i') + 'and' + str(
        adjThreshold2).replace('.', 'i') + '_g' + str(gamma).replace('.', 'i') + '_delta' + str(delta).replace(
        '.', 'i') + '_lda' + str(lda).replace('.', 'i') + '_ep' + str(ep).replace('.', 'i') + '_rounds' + str(
        trainingRounds) + '_shots' + str(shots) + '_plotmean.csv',
              index=False)
    # """


"""
    ### tested

trainingRounds=1000
shots=10

makeLossListMean([2, 3, 2], 4, 4, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"randomUnitary")
makeLossListMean([2, 3, 2], 4, 4, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"circleUnitary")
makeLossListMean([2, 3, 2], 4, 4, 1, 0.01, trainingRounds, -1, 0.5, 1, 0, shots,"lineUnitary")
makeLossListMean([2, 3, 2], 4, 4, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"ADC")
makeLossListMean([2, 1], 4, 4, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"classification")
makeLossListMean([2, 1], 4, 4, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"classificationSup")
makeLossListMean([2, 3, 2], 4, 4, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"circleOutput")
makeLossListMean([2, 3, 2], 4, 4, 1, 0.01, trainingRounds, -1, 0.5, 1, 0, shots,"lineOutput")
makeLossListMean([2, 1], 4, 4, 1, 0.01, trainingRounds, -1, 0.5, 1, 0, shots,"lineOutputRandom")
makeLossListMean([2, 1], 4, 4, 1, 0.01, trainingRounds, -1, 1, 1, 0, shots,"classificationSup") #has extra 1s in fidelity matrix
makeLossListMean([2, 1], 4, 4, 1, 0.01, trainingRounds, -1, 0.5, 1, 0, shots,"clustersRandom")
makeLossListMean([2, 1], 4, 4, 1, 0.01, trainingRounds, -1, 0.5, 1, 0, shots,"clustersRandomShuffled")
makeLossListMean([2, 1], 4, 4, 1, 0.01, trainingRounds, -1, 0.5, 1, 0, shots,"connectedClustersRandom")
makeLossListMean([2, 1], 4, 4, 1, 0.01, trainingRounds, -1, 0.5, 1, 0, shots,"connectedClustersRandomShuffled")
makeLossListMean([2, 1], 4, 4, 1, 0.01, trainingRounds, -1, 0.5, 1, 0, shots,"lineOutputRandomShuffled")

makeLossListMean([3, 4, 3], 8, 8, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"randomUnitary")
makeLossListMean([3, 4, 3], 8, 8, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"circleUnitary")
makeLossListMean([3, 4, 3], 8, 8, 1, 0.01, trainingRounds, -1, 0.89, 1, 0, shots,"lineUnitary")
makeLossListMean([3, 4, 3], 8, 8, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"ADC")
makeLossListMean([3, 1], 8, 8, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"classification")
makeLossListMean([3, 1], 8, 8, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"classificationSup")
makeLossListMean([3, 4, 3], 8, 8, 1, 0.01, trainingRounds, -1, 0.24, 1, 0, shots,"circleOutput")
makeLossListMean([3, 4, 3], 8, 8, 1, 0.01, trainingRounds, -1, 0.89, 1, 0, shots,"lineOutput")
makeLossListMean([3, 1], 8, 8, 1, 0.01, trainingRounds, -1, 0.89, 1, 0, shots,"lineOutputRandom")
makeLossListMean([3, 1], 8, 8, 1, 0.01, trainingRounds, -1, 1, 1, 0, shots,"classificationSup") #has extra 1s in fidelity matrix
makeLossListMean([3, 1], 8, 8, 1, 0.01, trainingRounds, -1, 0.89, 1, 0, shots,"clustersRandom")
makeLossListMean([3, 1], 8, 8, 1, 0.01, trainingRounds, -1, 0.89, 1, 0, shots,"clustersRandomShuffled")
makeLossListMean([3, 1], 8, 8, 1, 0.01, trainingRounds, -1, 0.65, 1, 0, shots,"connectedClustersRandom")
makeLossListMean([3, 1], 8, 8, 1, 0.01, trainingRounds, -1, 0.65, 1, 0, shots,"connectedClustersRandomShuffled")
makeLossListMean([3, 1], 8, 8, 1, 0.01, trainingRounds, -1, 0.89, 1, 0, shots,"lineOutputRandomShuffled")

makeLossListMean([4, 1], 16, 16, 1, 0.01, trainingRounds, -1, 0.74, 1, 0, shots,"connectedClustersRandom")
makeLossListMean([4, 1], 16, 16, 1, 0.01, trainingRounds, -1, 0.74, 1, 0, shots,"connectedClustersRandomShuffled")
makeLossListMean([4, 1], 16, 16, 1, 0.01, trainingRounds, -1, 0.74, 1, 0, shots,"clustersRandom")
makeLossListMean([4, 1], 16, 16, 1, 0.01, trainingRounds, -1, 0.74, 1, 0, shots,"clustersRandomShuffled")
makeLossListMean([4, 1], 16, 16, 1, 0.01, trainingRounds, -1, 1, 1, 0, shots,"classificationSup") #has extra 1s in fidelity matrix
makeLossListMean([4, 1], 16, 16, 1, 0.01, trainingRounds, -1, 0.98, 1, 0, shots,"lineOutputRandom")
makeLossListMean([4, 1], 16, 16, 1, 0.01, trainingRounds, -1, 0.98, 1, 0, shots,"lineOutputRandomShuffled")

trainingRounds=1000
shots=30
makeLossListMean([4, 1], 16, 16, 1, 0.01, trainingRounds, -1, 0.98, 1, 0, shots,"lineOutputRandomShuffled")
makeLossListMean([4, 1], 16, 16, 1, 0.01, trainingRounds, -1, 0.74, 1, 0, shots,"connectedClustersRandomShuffled")

"""
trainingRounds=1000
shots=30
gamma = -0.5

makeLossListMean([3, 1], 10, 10, 1, 0.01, trainingRounds, gamma, 0.93, 1, 0, shots,"lineOutputRandomShuffled")
makeLossListMean([3, 1], 8, 8, 1, 0.01, trainingRounds, gamma, 0.65, 1, 0, shots,"connectedClustersRandomShuffled")
"""
    ### not tested
"""


# to run the code use the function makeLossListMean with the following inputs:
# qnnArch: architecture of the network, for example [2,3,2] or [4,1]
# numTrainingPairs: number of training pairs
# maxNumberSupervisedPairs: the functions starts with one supervised training pair and ends with maxNumberSupervisedPairs supervised training pairs
# lda: lambda = 1/learning rate
# ep: epsilon = step size
# trainingRounds: training rounds for one shot
# gamma: negative number, weight of the graph part in the training cost
# adjThreshold: state pairs (ij), which a fidelity over adjThreshold are close (adjecency matrix A_{ij}=1)
# adjThreshold2: set this to 1 or use it: state pairs (ij) , which a fidelity over adjThreshold2 are a bit close (adjecency matrix A_{ij}=0.5)
# delta: set to 0 or use it: creates noise in the training pairs
# shots: number of shots for mean values in the plot
# kind: the kind of the graph choose from: ADC, circleOutput, circleUnitary, classification, classificationSup, clustersRandom, clustersRandomShuffled, connectedClustersRandom, connectedClustersRandomShuffled, lineOutput, lineOutputRandom, lineUnitary, randomUnitary
package utils

import (
	"log"

	"github.com/aintsashqa/go-simple-neural-network-lib/network"
	"github.com/aintsashqa/go-simple-neural-network-lib/types"
)

type Balancer struct {
	Dataset func() ([]struct {
		NeuronIndex int
		NeuronValue float64
	}, [][]float64)
	EpochCount   uint32
	LearningRate float64
}

func Balance(net *network.NeuralNetwork, balancer *Balancer) {
	if net == nil || balancer == nil {
		return
	}

	expecteds, inputs := balancer.Dataset()
	if len(expecteds) != len(inputs) {
		return
	}

	balanceError := 0.0

	for currentEpoch := 0; currentEpoch < int(balancer.EpochCount); currentEpoch++ {
		for index := 0; index < len(expecteds); index++ {
			balanceError += balance_backPropagationNetwork(net, balancer.LearningRate, expecteds[index], inputs[index])
		}
		if currentEpoch%1000 == 0 {
			log.Printf("[utils.Balance]: Balance error %f on epoch %d", balanceError/float64(currentEpoch), currentEpoch)
		}
	}

	log.Printf("[utils.Balance]: Balance error result %f", balanceError/float64(balancer.EpochCount))
}

func balance_backPropagationNetwork(net *network.NeuralNetwork, rate float64, expected struct {
	NeuronIndex int
	NeuronValue float64
}, signals []float64) float64 {
	_, actualValue := net.FeedForward(signals)
	difference := actualValue - expected.NeuronValue

	current := net.Layers.Back()
	if current == nil {
		return 0
	}

	previous := current.Prev()
	if previous == nil {
		return 0
	}

	previousSignals := previous.Value.(*network.Layer).GetOutputSignals()
	for _, neuron := range current.Value.(*network.Layer).Neurons {
		balance_backPropagationNeuron(neuron, difference, rate, previousSignals)
	}

	next := previous.Next()
	for current = previous; current != nil; current = current.Prev() {
		if current.Prev() == nil {
			break
		}

		previous = current.Prev()
		previousSignals = previous.Value.(*network.Layer).GetOutputSignals()
		for index, neuron := range current.Value.(*network.Layer).Neurons {
			for _, nextNeuron := range next.Value.(*network.Layer).Neurons {
				nextNeuronDifference := nextNeuron.Weights[index] * nextNeuron.Delta
				balance_backPropagationNeuron(neuron, nextNeuronDifference, rate, previousSignals)
			}
		}

		next = next.Prev()
	}

	return difference * difference
}

func balance_backPropagationNeuron(currentNeuron *network.Neuron, difference, rate float64, previousSignals []float64) {
	if currentNeuron.Type == types.NeuronType_Input {
		return
	}

	funcResult := currentNeuron.ActivationFunc(currentNeuron.Output)
	currentNeuron.Delta = difference * (funcResult * (1 - funcResult))

	for index := 0; index < len(currentNeuron.Weights); index++ {
		previousSignalValue := previousSignals[index]
		nextWeight := currentNeuron.Weights[index] - previousSignalValue*currentNeuron.Delta*rate
		currentNeuron.Weights[index] = nextWeight
	}
}

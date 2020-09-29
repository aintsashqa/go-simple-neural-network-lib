package network

import (
	"container/list"
	"log"

	"github.com/aintsashqa/go-simple-neural-network-lib/functions"
	"github.com/aintsashqa/go-simple-neural-network-lib/types"
)

type NeuralNetwork struct {
	Options *NeuralNetworkOptions

	Layers *list.List
}

func NewNeuralNetwork(options *NeuralNetworkOptions) *NeuralNetwork {
	network := new(NeuralNetwork)

	if options == nil {
		options = &NeuralNetworkOptions{
			InputLayerNeuronsCount:   3,
			HiddenLayersNeuronsCount: []uint32{2},
			OutputLayerNeuronsCount:  1,

			ActivationFunc: functions.Sigmoid,
		}
	}

	network.Options = options
	network.Layers = list.New()
	network.init()

	return network
}

func (n *NeuralNetwork) init() {
	inputLayer := n.createInputLayer()
	n.Layers.PushBack(inputLayer)

	for _, count := range n.Options.HiddenLayersNeuronsCount {
		hiddenLayer := n.createHiddenLayer(count)
		n.Layers.PushBack(hiddenLayer)
	}

	outputLayer := n.createOutputLayer()
	n.Layers.PushBack(outputLayer)
}

func (n *NeuralNetwork) createLayer(signalsCount, neuronsCount uint32, neuronsType types.NeuronType, activationFunc types.ActivationFunc) *Layer {
	return NewLayer(signalsCount, neuronsCount, neuronsType, activationFunc)
}

func (n *NeuralNetwork) createInputLayer() *Layer {
	return n.createLayer(1, n.Options.InputLayerNeuronsCount, types.NeuronType_Input, n.Options.ActivationFunc)
}

func (n *NeuralNetwork) createHiddenLayer(neuronsCount uint32) *Layer {
	lastLayer := n.Layers.Back().Value.(*Layer)
	return n.createLayer(uint32(len(lastLayer.Neurons)), neuronsCount, types.NeuronType_Hidden, n.Options.ActivationFunc)
}

func (n *NeuralNetwork) createOutputLayer() *Layer {
	lastLayer := n.Layers.Back().Value.(*Layer)
	return n.createLayer(uint32(len(lastLayer.Neurons)), n.Options.OutputLayerNeuronsCount, types.NeuronType_Output, n.Options.ActivationFunc)
}

func (n *NeuralNetwork) FeedForward(signals []float64) (int, float64) {
	first := n.Layers.Front()
	if first == nil {
		log.Print("[network.*NeuralNetwork.FeedForward]: Input layer equals nil")
		return 0, 0
	}

	inputLayer := first.Value.(*Layer)
	var nextSignals []float64
	for index, signal := range signals {
		output := inputLayer.Neurons[index].Feed([]float64{signal})
		nextSignals = append(nextSignals, output)
	}

	for next := first.Next(); next != nil; next = next.Next() {
		layer := next.Value.(*Layer)
		temp := layer.FeedNeurons(nextSignals)
		nextSignals = temp
	}

	var index int = 0
	var result float64 = 0.0
	if n.Options.OutputLayerNeuronsCount == 1 {
		result = n.Layers.Back().Value.(*Layer).Neurons[index].Output
	} else {
		for currentIndex, neuron := range n.Layers.Back().Value.(*Layer).Neurons {
			if neuron.Output > result {
				result = neuron.Output
				index = currentIndex
			}
		}
	}

	return index, result
}

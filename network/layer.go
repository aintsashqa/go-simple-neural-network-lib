package network

import "github.com/aintsashqa/go-simple-neural-network-lib/types"

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(signalsCount, neuronsCount uint32, neuronsType types.NeuronType, activationFunc types.ActivationFunc) *Layer {
	layer := new(Layer)
	layer.init(signalsCount, neuronsCount, neuronsType, activationFunc)

	return layer
}

func (l *Layer) init(signalsCount, neuronsCount uint32, neuronsType types.NeuronType, activationFunc types.ActivationFunc) {
	for index := 0; index < int(neuronsCount); index++ {
		l.Neurons = append(l.Neurons, NewNeuron(signalsCount, neuronsType, activationFunc))
	}
}

func (l *Layer) FeedNeurons(signals []float64) []float64 {
	var outputSignals []float64

	for _, neuron := range l.Neurons {
		output := neuron.Feed(signals)
		outputSignals = append(outputSignals, output)
	}

	return outputSignals
}

func (l *Layer) GetOutputSignals() []float64 {
	var signals []float64

	for _, neuron := range l.Neurons {
		signals = append(signals, neuron.Output)
	}

	return signals
}

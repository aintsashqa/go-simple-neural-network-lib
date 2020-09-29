package network

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/aintsashqa/go-simple-neural-network-lib/types"
)

type Neuron struct {
	Type           types.NeuronType
	ActivationFunc types.ActivationFunc

	Weights []float64
	Output  float64
	Delta   float64
}

func NewNeuron(signalsCount uint32, neuronType types.NeuronType, activationFunc types.ActivationFunc) *Neuron {
	neuron := new(Neuron)
	neuron.Type = neuronType
	neuron.ActivationFunc = activationFunc

	neuron.init(signalsCount)

	return neuron
}

func (n *Neuron) init(signalsCount uint32) {
	for index := 0; index < int(signalsCount); index++ {
		value := 1.0

		if n.Type != types.NeuronType_Input {
			value = rand.Float64()
		}

		n.Weights = append(n.Weights, value)
	}
}

func (n *Neuron) Feed(signals []float64) float64 {
	if len(signals) != len(n.Weights) {
		log.Printf("[network.*Neuron.Feed]: Neuron weights count (%d) not equals signals count (%d)", len(n.Weights), len(signals))
		return 0
	}

	var weightsSum float64
	for index, signal := range signals {
		weightsSum += signal * n.Weights[index]
	}

	if n.Type != types.NeuronType_Input {
		if n.ActivationFunc == nil {
			log.Print("[network.*Neuron.Feed]: Neuron activation func equals nil")
			return 0
		}

		n.Output = n.ActivationFunc(weightsSum)
	} else {
		n.Output = weightsSum
	}

	return n.Output
}

func (n *Neuron) String() string {
	return fmt.Sprintf("%f", n.Output)
}

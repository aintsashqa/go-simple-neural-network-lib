package network_test

import (
	"testing"

	"github.com/aintsashqa/go-simple-neural-network-lib/network"
	"github.com/aintsashqa/go-simple-neural-network-lib/types"
	"github.com/stretchr/testify/assert"
)

func activationFunc(value float64) float64 {
	return value
}

func createNeurons(activationFunc types.ActivationFunc) (*network.Neuron, *network.Neuron, *network.Neuron) {
	inputNeuron := network.NewNeuron(1, types.NeuronType_Input, activationFunc)

	hiddenNeuron := network.NewNeuron(1, types.NeuronType_Hidden, activationFunc)
	hiddenNeuron.Weights[0] = 1.0

	outputNeuron := network.NewNeuron(1, types.NeuronType_Output, activationFunc)
	outputNeuron.Weights[0] = 1.0

	return inputNeuron, hiddenNeuron, outputNeuron
}

func TestNewNeuronSuccess(t *testing.T) {
	a := assert.New(t)

	inputNeuron, hiddenNeuron, outputNeuron := createNeurons(activationFunc)

	a.NotNil(inputNeuron)
	a.Equal(types.NeuronType_Input, inputNeuron.Type)
	a.Equal(1, len(inputNeuron.Weights))
	a.Equal(1.0, inputNeuron.Weights[0])
	a.Equal(0.0, inputNeuron.Output)

	a.NotNil(hiddenNeuron)
	a.Equal(types.NeuronType_Hidden, hiddenNeuron.Type)
	a.NotNil(hiddenNeuron.ActivationFunc)
	a.Equal(1, len(hiddenNeuron.Weights))
	a.Equal(0.0, hiddenNeuron.Output)

	a.NotNil(outputNeuron)
	a.Equal(types.NeuronType_Output, outputNeuron.Type)
	a.NotNil(outputNeuron.ActivationFunc)
	a.Equal(1, len(outputNeuron.Weights))
	a.Equal(0.0, outputNeuron.Output)
}

func TestNeuronFeedSuccess(t *testing.T) {
	a := assert.New(t)

	inputNeuron, hiddenNeuron, outputNeuron := createNeurons(activationFunc)

	expected := 1.0
	signals := []float64{
		1.0,
	}

	inputActual := inputNeuron.Feed(signals)
	hiddenActual := hiddenNeuron.Feed(signals)
	outputActual := outputNeuron.Feed(signals)

	a.Equal(expected, inputActual)
	a.Equal(expected, hiddenActual)
	a.Equal(expected, outputActual)
}

func TestNeuronFeedFailedSignalsCount(t *testing.T) {
	a := assert.New(t)

	inputNeuron, hiddenNeuron, outputNeuron := createNeurons(activationFunc)

	expected := 0.0
	signals := []float64{
		1.0, 2.0,
	}

	inputActual := inputNeuron.Feed(signals)
	hiddenActual := hiddenNeuron.Feed(signals)
	outputActual := outputNeuron.Feed(signals)

	a.Equal(expected, inputActual)
	a.Equal(expected, hiddenActual)
	a.Equal(expected, outputActual)
}

func TestNeuronFeedFailedActivationFunc(t *testing.T) {
	a := assert.New(t)

	_, hiddenNeuron, outputNeuron := createNeurons(nil)

	expected := 0.0
	signals := []float64{
		1.0,
	}

	hiddenActual := hiddenNeuron.Feed(signals)
	outputActual := outputNeuron.Feed(signals)

	a.Equal(expected, hiddenActual)
	a.Equal(expected, outputActual)
}

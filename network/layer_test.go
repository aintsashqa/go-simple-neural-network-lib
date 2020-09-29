package network_test

import (
	"testing"

	"github.com/aintsashqa/go-simple-neural-network-lib/network"
	"github.com/aintsashqa/go-simple-neural-network-lib/types"
	"github.com/stretchr/testify/assert"
)

func TestNewLayerSuccess(t *testing.T) {
	a := assert.New(t)

	neuronsCount := 10
	layer := network.NewLayer(1, uint32(neuronsCount), types.NeuronType_Input, nil)

	a.NotNil(layer)
	a.Equal(neuronsCount, len(layer.Neurons))
}

func TestLayerFeedNeuronSuccess(t *testing.T) {
	a := assert.New(t)

	expected := []float64{
		1.0, 1.0, 1.0,
	}
	signals := []float64{
		1.0,
	}

	layer := network.NewLayer(uint32(len(signals)), uint32(len(expected)), types.NeuronType_Input, activationFunc)

	actual := layer.FeedNeurons(signals)

	a.Equal(expected, actual)
}

func TestLayerGetOutputSignalsSuccess(t *testing.T) {
	a := assert.New(t)

	expected := []float64{
		0.0, 0.0, 0.0,
	}

	layer := network.NewLayer(1, uint32(len(expected)), types.NeuronType_Input, activationFunc)

	actual := layer.GetOutputSignals()

	a.Equal(expected, actual)
}

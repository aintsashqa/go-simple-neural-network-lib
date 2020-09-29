package network_test

import (
	"testing"

	"github.com/aintsashqa/go-simple-neural-network-lib/network"
	"github.com/stretchr/testify/assert"
)

func TestNewNeuralNetworkSuccess(t *testing.T) {
	a := assert.New(t)

	options := network.NeuralNetworkOptions{
		InputLayerNeuronsCount:   3,
		HiddenLayersNeuronsCount: []uint32{2},
		OutputLayerNeuronsCount:  1,
	}
	net := network.NewNeuralNetwork(&options)

	a.NotNil(net)
	a.Equal(3, net.Layers.Len())
}

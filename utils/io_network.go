package utils

import (
	"github.com/aintsashqa/go-simple-neural-network-lib/network"
)

type (
	io_NeuralNetworkLayer struct {
		Layer *network.Layer
	}

	io_NeuralNetwork struct {
		Layers  []io_NeuralNetworkLayer
		Options network.NeuralNetworkOptions
	}
)

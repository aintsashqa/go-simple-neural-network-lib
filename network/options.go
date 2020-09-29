package network

import (
	"github.com/aintsashqa/go-simple-neural-network-lib/types"
)

type NeuralNetworkOptions struct {
	InputLayerNeuronsCount   uint32
	HiddenLayersNeuronsCount []uint32
	OutputLayerNeuronsCount  uint32

	ActivationFunc types.ActivationFunc
}

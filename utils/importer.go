package utils

import (
	"bytes"
	"container/list"
	"encoding/gob"
	"io/ioutil"

	"github.com/aintsashqa/go-simple-neural-network-lib/functions"
	"github.com/aintsashqa/go-simple-neural-network-lib/network"
)

func Import(filename string) (*network.NeuralNetwork, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	net, err := import_Decode(data)
	if err != nil {
		return nil, err
	}

	return net, nil
}

func import_Decode(data []byte) (*network.NeuralNetwork, error) {
	importStructure := io_NeuralNetwork{}

	buffer := bytes.Buffer{}
	buffer.Write(data)
	decoder := gob.NewDecoder(&buffer)

	err := decoder.Decode(&importStructure)
	if err != nil {
		return nil, err
	}

	// TODO: Find way to encode and decode activation func
	net := new(network.NeuralNetwork)
	net.Options = &importStructure.Options
	net.Options.ActivationFunc = functions.Sigmoid
	net.Layers = list.New()
	for _, io_Layer := range importStructure.Layers {
		for _, neuron := range io_Layer.Layer.Neurons {
			neuron.ActivationFunc = functions.Sigmoid
		}
		net.Layers.PushBack(io_Layer.Layer)
	}

	return net, nil
}

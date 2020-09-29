package utils

import (
	"bytes"
	"encoding/gob"
	"os"

	"github.com/aintsashqa/go-simple-neural-network-lib/network"
)

func Export(net *network.NeuralNetwork, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	data, err := export_Encode(net)
	if err != nil {
		return err
	}

	file.Write(data)

	return nil
}

func export_Encode(net *network.NeuralNetwork) ([]byte, error) {
	exportStructure := io_NeuralNetwork{}

	for current := net.Layers.Front(); current != nil; current = current.Next() {
		exportLayer := io_NeuralNetworkLayer{}
		exportLayer.Layer = current.Value.(*network.Layer)
		exportStructure.Layers = append(exportStructure.Layers, exportLayer)
	}

	buffer := bytes.Buffer{}
	encoder := gob.NewEncoder(&buffer)

	if err := encoder.Encode(exportStructure); err != nil {
		return nil, err
	}

	return buffer.Bytes(), nil
}

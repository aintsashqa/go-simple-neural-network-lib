package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/aintsashqa/go-simple-neural-network-lib/functions"
	"github.com/aintsashqa/go-simple-neural-network-lib/network"
	"github.com/aintsashqa/go-simple-neural-network-lib/utils"
)

var (
	filename *string = flag.String("filename", "", "Write filename to import neural network data")
)

func init() {
	flag.Parse()
}

func main() {
	var net *network.NeuralNetwork

	currentDir, _ := os.Getwd()
	currentDir = fmt.Sprintf("%s/examples/base", currentDir)
	if *filename == "" {
		options := network.NeuralNetworkOptions{
			InputLayerNeuronsCount:   4,
			HiddenLayersNeuronsCount: []uint32{2},
			OutputLayerNeuronsCount:  1,
			ActivationFunc:           functions.Sigmoid,
		}

		net = network.NewNeuralNetwork(&options)

		balancer := utils.Balancer{
			EpochCount:   1000000,
			LearningRate: 0.01,
			Dataset:      dataset,
		}

		utils.Balance(net, &balancer)

		*filename = fmt.Sprintf("%s/net", currentDir)
		if err := utils.Export(net, *filename); err != nil {
			log.Print(err)
		}
	} else {
		var err error
		*filename = fmt.Sprintf("%s/%s", currentDir, *filename)
		net, err = utils.Import(*filename)
		if err != nil {
			log.Fatal(err)
		}
	}

	_, value := net.FeedForward([]float64{
		1, 0, 0, 1, // 10
	})

	fmt.Println(value)
}

func dataset() ([]struct {
	NeuronIndex int
	NeuronValue float64
}, [][]float64) {
	return []struct {
			NeuronIndex int
			NeuronValue float64
		}{
			{NeuronIndex: 0, NeuronValue: 0}, // 1
			{NeuronIndex: 0, NeuronValue: 0}, // 2
			{NeuronIndex: 0, NeuronValue: 1}, // 3
			{NeuronIndex: 0, NeuronValue: 0}, // 4
			{NeuronIndex: 0, NeuronValue: 0}, // 5
			{NeuronIndex: 0, NeuronValue: 0}, // 6
			{NeuronIndex: 0, NeuronValue: 1}, // 7
			{NeuronIndex: 0, NeuronValue: 0}, // 8
			{NeuronIndex: 0, NeuronValue: 1}, // 9
			{NeuronIndex: 0, NeuronValue: 1}, // 10
			{NeuronIndex: 0, NeuronValue: 1}, // 11
			{NeuronIndex: 0, NeuronValue: 1}, // 12
			{NeuronIndex: 0, NeuronValue: 1}, // 13
			{NeuronIndex: 0, NeuronValue: 0}, // 14
			{NeuronIndex: 0, NeuronValue: 1}, // 15
			{NeuronIndex: 0, NeuronValue: 1}, // 16
		}, [][]float64{
			{0, 0, 0, 0}, // 1
			{0, 0, 0, 1}, // 2
			{0, 0, 1, 0}, // 3
			{0, 0, 1, 1}, // 4
			{0, 1, 0, 0}, // 5
			{0, 1, 0, 1}, // 6
			{0, 1, 1, 0}, // 7
			{0, 1, 1, 1}, // 8
			{1, 0, 0, 0}, // 9
			{1, 0, 0, 1}, // 10
			{1, 0, 1, 0}, // 11
			{1, 0, 1, 1}, // 12
			{1, 1, 0, 0}, // 13
			{1, 1, 0, 1}, // 14
			{1, 1, 1, 0}, // 15
			{1, 1, 1, 1}, // 16
		}
}

package main

import (
	"fmt"

	"github.com/aintsashqa/go-simple-neural-network-lib/functions"
	"github.com/aintsashqa/go-simple-neural-network-lib/network"
	"github.com/aintsashqa/go-simple-neural-network-lib/utils"
)

func main() {
	options := network.NeuralNetworkOptions{
		InputLayerNeuronsCount:   4,
		HiddenLayersNeuronsCount: []uint32{2},
		OutputLayerNeuronsCount:  1,

		ActivationFunc: functions.Sigmoid,
	}

	net := network.NewNeuralNetwork(&options)

	balancer := utils.Balancer{
		Dataset:      dataset,
		EpochCount:   100000,
		LearningRate: 0.01,
	}

	utils.Balance(net, &balancer)

	result, inputs := dataset()

	fmt.Println("Expected\t#\tPredict")
	for index, r := range result {
		_, output := net.FeedForward(inputs[index])
		fmt.Printf("%.3f\t\t#\t%.3f\n", r.NeuronValue, output)
	}
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

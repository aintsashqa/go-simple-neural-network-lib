package types

type NeuronType uint8

const (
	NeuronType_Input  NeuronType = 0
	NeuronType_Hidden NeuronType = 1
	NeuronType_Output NeuronType = 2
)

func (nt NeuronType) String() string {
	switch nt {
	case NeuronType_Input:
		return "InputType"
	case NeuronType_Hidden:
		return "HiddenType"
	case NeuronType_Output:
		return "OutputType"

	default:
		return "Invalid neuron type"
	}
}

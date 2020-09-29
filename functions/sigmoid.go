package functions

import (
	"math"
)

func Sigmoid(value float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -value))
}

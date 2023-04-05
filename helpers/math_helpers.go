package helpers

import (
	"math"
)

func Sigmoid(r, c int, x float64) float64 {
	return 1/(1 + math.Exp((-1) * x))
}
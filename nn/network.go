package nn

import (
	"dl-lib/helpers"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"math"
	"os"
)

type Network struct {
	inputs        int
	hiddens       int
	outputs       int
	hiddenWeights *mat.Dense
	outputWeights *mat.Dense
	learningRate  float64
}
func randomArray(size int, v float64) (data []float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(v),
		Max: 1 / math.Sqrt(v),
	}

	data = make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = dist.Rand()
	}
	return
}
func CreateNetwork(input, hidden, output int, rate float64) (net Network) {
	net = Network{
		inputs:       input,
		hiddens:      hidden,
		outputs:      output,
		learningRate: rate,
	}

	net.hiddenWeights = mat.NewDense(net.hiddens, net.inputs, randomArray(net.inputs*net.hiddens, float64(net.inputs)))
	net.outputWeights = mat.NewDense(net.outputs, net.hiddens, randomArray(net.inputs*net.hiddens, float64(net.inputs)))

	return
}

func (net Network)forwardPropagation(inputData []float64) mat.Matrix{
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := helpers.Dot(net.hiddenWeights, inputs)
	hiddenOutputs := helpers.Apply(helpers.Sigmoid, hiddenInputs)
	finalInputs := helpers.Dot(net.outputWeights, hiddenOutputs)
	finalOutputs := helpers.Apply(helpers.Sigmoid, finalInputs)

	return finalOutputs
}
func (net Network)Predict(inputData []float64) mat.Matrix{
	return net.forwardPropagation(inputData)
}

func (net Network)Train(inputData []float64, targetData []float64) {
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := helpers.Dot(net.hiddenWeights, inputs)
	hiddenOutputs := helpers.Apply(helpers.Sigmoid, hiddenInputs)
	finalInputs := helpers.Dot(net.outputWeights, hiddenOutputs)
	finalOutputs := helpers.Apply(helpers.Sigmoid, finalInputs)

	//Errors
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := helpers.Subtract(targets, finalOutputs)
	hiddenErrors := helpers.Dot(net.outputWeights.T(), outputErrors)

	//Back propagation

	net.outputWeights = helpers.Add(net.outputWeights,
		helpers.Scale(net.learningRate,
			helpers.Dot(helpers.Multiply(outputErrors, helpers.SigmoidPrime(finalOutputs)),
				hiddenOutputs.T()))).(*mat.Dense)

	net.hiddenWeights = helpers.Add(net.hiddenWeights,
		helpers.Scale(net.learningRate,
			helpers.Dot(helpers.Multiply(hiddenErrors, helpers.SigmoidPrime(hiddenOutputs)),
				inputs.T()))).(*mat.Dense)

}

func save(net Network, hWeightsFileName string, oWeightsFileName string) {
	h, err := os.Create(hWeightsFileName)
	defer h.Close()
	if err == nil {
		net.hiddenWeights.MarshalBinaryTo(h)
	}
	o, err := os.Create(oWeightsFileName)
	defer o.Close()
	if err == nil {
		net.outputWeights.MarshalBinaryTo(o)
	}
}

// load a neural network from file
func load(net *Network, hWeightsFileName string, oWeightsFileName string) {
	h, err := os.Open(hWeightsFileName)
	defer h.Close()
	if err == nil {
		net.hiddenWeights.Reset()
		net.hiddenWeights.UnmarshalBinaryFrom(h)
	}
	o, err := os.Open(oWeightsFileName)
	defer o.Close()
	if err == nil {
		net.outputWeights.Reset()
		net.outputWeights.UnmarshalBinaryFrom(o)
	}
	return
}
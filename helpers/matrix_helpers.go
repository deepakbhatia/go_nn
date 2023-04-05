package helpers

import "gonum.org/v1/gonum/mat"

//Dot Wrapper
func Dot(m, n mat.Matrix) mat.Matrix {
	r,_ := m.Dims()
	_,c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func Apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r,c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func Scale(s float64, m mat.Matrix) mat.Matrix {
	r,c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, o)
	return o
}

func Multiply(n, m mat.Matrix) mat.Matrix {
	r,c :=n.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(n, m)
	return o
}
func Add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func Subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func AddScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x:=0; x<r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return Add(m, n)
}
//Derivate of the Sigmoid function
func SigmoidPrime(m mat.Matrix) mat.Matrix {
	rows, _ := m.Dims()
	o := make([]float64, rows)
	for i := range o {
		o[i] = 1
	}
	ones := mat.NewDense(rows, 1, o)
	return Multiply(m, Subtract(ones, m)) // m * (1 - m)
}

package activations;

import org.jblas.FloatMatrix;

public class Relu implements Activation {

	public FloatMatrix forward(FloatMatrix x) {
		for (int i=0; i<x.length; i++) {
			x.data[i] = Math.max(0, x.data[i]); 
		}
		return x;
	}

	public FloatMatrix backward(FloatMatrix dy, FloatMatrix preY, FloatMatrix y) {
		for (int i=0; i<y.length; i++) {
			dy.data[i] *= y.data[i] > 0 ? 1 : 0; 
		}
		return dy;
	}
}

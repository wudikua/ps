package activations;



import org.jblas.FloatMatrix;

public class Sigmoid implements Activation {

	public FloatMatrix forward(FloatMatrix x) {
		for (int i=0; i<x.length; i++) {
			x.data[i] = (float) (0.001f + (.999f-0.001f) / (1f + Math.exp(-x.data[i])));
		}
		return x;
	}

	public FloatMatrix backward(FloatMatrix dy, FloatMatrix preY, FloatMatrix y) {
		for (int i=0; i<y.length; i++) {
			dy.data[i] *= y.data[i] * (1 - y.data[i]); 
		}
		return dy;
	}
}

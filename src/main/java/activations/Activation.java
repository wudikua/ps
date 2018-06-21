package activations;


import org.jblas.FloatMatrix;

public interface Activation {

	public FloatMatrix forward(FloatMatrix x);

	public FloatMatrix backward(FloatMatrix dy, FloatMatrix preY, FloatMatrix y);
}

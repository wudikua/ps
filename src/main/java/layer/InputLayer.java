package layer;


import org.jblas.FloatMatrix;

public class InputLayer extends Layer {

	public InputLayer(String name, int inputDims, int outputDims) {
		super(name, inputDims, outputDims);
	}

	@Override
	public FloatMatrix forward() {
		return A;
	}

	@Override
	public FloatMatrix backward() {
		return next.delta;
	}

	@Override
	public void pullWeights() {
	}
}

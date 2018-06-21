package loss;

import org.jblas.FloatMatrix;

public interface Loss {

	public float forward(FloatMatrix predict, FloatMatrix label);

	public FloatMatrix backward(FloatMatrix predict, FloatMatrix label);
}

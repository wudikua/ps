package loss;

import org.apache.commons.math3.util.FastMath;
import org.jblas.FloatMatrix;

public class CrossEntropy implements Loss {
	
	public static final float slim = (float) Math.pow(10, -2);

	public float forward(FloatMatrix predict, FloatMatrix label) {
		float sum = 0;
		for (int i=0; i<predict.columns; i++) {
			float p =  predict.get(0, i);
			float l =  label.get(0, i);
			sum += (float) (-l * FastMath.log(p) - ((1 - l) * FastMath.log(1 - p)));
		}
		return sum / predict.columns;
	}

	public FloatMatrix backward(FloatMatrix predict, FloatMatrix label) {
		FloatMatrix delta = predict.dup();
		for (int i=0; i<predict.columns; i++) {
			float p =  predict.get(0, i);
			float l =  label.get(0, i);
			delta.data[predict.index(0, i)] = (p - l) / (p * (1 - p));
		}
		return delta;
	}
}

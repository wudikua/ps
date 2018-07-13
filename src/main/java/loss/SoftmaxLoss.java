package loss;

import org.apache.commons.math3.util.FastMath;
import org.jblas.FloatMatrix;

public class SoftmaxLoss implements Loss {

	// label is one-hot
	public float forward(FloatMatrix predict, FloatMatrix label) {
		float sum = 0;
		for (int i=0; i<predict.columns; i++) {
			int hot =  (int)label.get(0, i);
			float p =  predict.get(hot, i);
			sum += -FastMath.log(p);
		}
		return sum / predict.columns;
	}

	// label is one-hot
	public FloatMatrix backward(FloatMatrix predict, FloatMatrix label) {
		FloatMatrix delta = FloatMatrix.zeros(predict.rows, predict.columns);
		for (int i=0; i<predict.columns; i++) {
			int hot =  (int)label.get(0, i);
			float p =  predict.get(hot, i);
			delta.data[predict.index(hot, i)] = -1/p;
		}
		return delta;
	}
}

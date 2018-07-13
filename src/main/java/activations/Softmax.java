package activations;

import org.apache.commons.math3.util.FastMath;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

public class Softmax implements Activation {

	private int scale;

	public Softmax() {
		scale = 10000;
	}

	public Softmax(int scale) {
		this.scale = scale;
	}

	// x有多行是多分类结果 列是batch数量
	//
	public FloatMatrix forward(FloatMatrix x) {
		x.divi(scale);
		// shift
		FloatMatrix shift = x.columnMaxs();
		// exp
		for (int i=0; i<x.length; i++) {
			x.data[i] = (float)Math.exp(x.data[i] - shift.get(i / x.rows));
		}
		// row sum
		float[] sum = new float[x.columns];
		for (int i=0; i<x.columns; i++) {
			sum[i] = x.getColumn(i).sum();
		}
		for (int i=0; i<x.length; i++) {
			x.data[i] = x.data[i] / sum[i / x.rows];
			if (x.data[i] == 0) {
				x.data[i] = 0.001f;
			} else if (x.data[i] == 1) {
				x.data[i] = 0.999f;
			}
		}
		return x;
	}

	public FloatMatrix backward(FloatMatrix dy, FloatMatrix preY, FloatMatrix y) {
		// each sample
		FloatMatrix delta = FloatMatrix.zeros(y.rows, y.columns);
		for (int i=0; i<y.columns; i++) {
			// each class error
			for (int j=0;j<y.rows; j++) {
				if (dy.data[dy.index(j, i)] == 0) {
					continue;
				}
				float d = dy.data[dy.index(j, i)];
				// each gradient
				for (int k=0; k<y.rows; k++) {
					if (j == k) {
						delta.data[dy.index(k, i)] += y.data[y.index(k, i)] * (1-y.data[y.index(k, i)]);
					} else {
						delta.data[dy.index(k, i)] += - y.data[y.index(j, i)] * y.data[y.index(k, i)];
					}
					delta.data[dy.index(k, i)] *= d;
				}
			}
		}
		return delta;
	}
}

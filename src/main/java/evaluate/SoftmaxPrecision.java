package evaluate;


import org.jblas.FloatMatrix;

import java.util.List;

public class SoftmaxPrecision {
	float[][] p;
	float[] l;

	public SoftmaxPrecision(FloatMatrix l, FloatMatrix p) {
		this.l = l.toArray();
		this.p = new float[p.columns][];
		for (int i=0; i<p.columns; i++) {
			this.p[i] = new float[p.rows];
			for (int j=0; j<p.rows; j++) {
				this.p[i][j] = p.data[p.index(j, i)];
			}
		}
	}

	public SoftmaxPrecision(List<Float> l, List<Float[]> p) {
		this.l = new float[l.size()];
		this.p = new float[p.size()][];
		for (int i=0; i<l.size(); i++) {
			this.l[i] = l.get(i);
			this.p[i] = new float[p.get(i).length];
			for (int j=0; j<p.get(i).length; j++) {
				this.p[i][j] = p.get(i)[j];
			}
		}
	}

	public SoftmaxPrecision(float[][] p, float[] l) {
		this.p = p;
		this.l = l;
	}

	public float calculate() {
		float yes = 0;
		// each sample
		for (int i=0; i<l.length; i++) {
			if (l[i] == max(p[i])) {
				yes ++;
			}
		}
		return yes/l.length;
	}

	private int max(float[] c) {
		int m = 0;
		float max = c[0];
		for (int i=0; i<c.length; i++) {
			if (c[i] > max) {
				m = i;
				max = c[i];
			}
		}
		return m;
	}
}

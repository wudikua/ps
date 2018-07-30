package layer;


import context.Context;
import org.jblas.FloatMatrix;
import util.MatrixUtil;

public class DropoutLayer extends Layer {

	float p = 0.5f;

	FloatMatrix mask;

	boolean scale;

	public DropoutLayer(String name, float p, boolean scale) {
		this.name = name;
		this.p = p;
		this.scale = scale;
	}

	@Override
	public FloatMatrix forward() {
		FloatMatrix A = this.pre.A;
		this.A = A;
		if (Context.isTraining()) {
			// 只有训练阶段dropout才起作用
			mask = MatrixUtil.randBernoulli(A.rows, A.columns, 1 - p);
			this.A = A.muli(mask);
			if (scale) {
				this.A = this.A.divi(1 - p);
			}
		}
		return this.A;
	}

	@Override
	public FloatMatrix backward() {
		FloatMatrix delta;
		if (next == null) {
			delta = this.delta;
		} else {
			delta = next.delta;
		}
		this.delta = delta;
		if (Context.isTraining()) {
			this.delta = delta.muli(mask);
			if (scale) {
				this.delta = this.delta.divi(1 - p);
			}
		}
		return this.delta;
	}

	@Override
	public void pullWeights() {
		mask = null;
		return;
	}
}

import activations.Relu;
import context.Context;
import layer.Conv2DLayer;
import layer.InputLayer;
import layer.PoolingLayer;
import org.jblas.FloatMatrix;
import org.junit.Test;
import util.MatrixUtil;

public class TestConv {

	@Test
	public void testImg2Col() {
		Context.init();
		FloatMatrix f = FloatMatrix.zeros(9, 2);
		int k = 1;
		for (int i=0; i<f.rows; i++) {
			for (int j=0; j<f.columns; j++) {
				f.data[f.index(i, j)] = k++;
			}
		}
		System.out.println("origin " + MatrixUtil.pretty(f));
		Conv2DLayer conv = new Conv2DLayer("conv", 3, 3, 1, 2, 1, 2, 0 );
		conv.setPadding(0);
		FloatMatrix r = conv.im2col(f);
		System.out.println("img col " + MatrixUtil.pretty(r));
		FloatMatrix c = conv.col2im(r, 2);
		System.out.println("origin " + MatrixUtil.pretty(c));
	}

	@Test
	public void testConv() {
		Context.init();
		FloatMatrix f = FloatMatrix.zeros(9, 2);
		int k = 1;
		for (int i=0; i<f.rows; i++) {
			for (int j=0; j<f.columns; j++) {
				f.data[f.index(i, j)] = k++;
			}
		}
		System.out.println("origin " + MatrixUtil.pretty(f));
		Conv2DLayer conv = new Conv2DLayer("conv", 3, 3, 1, 2, 1, 2, 0);
		conv.setPadding(0);
		FloatMatrix r = conv.im2col(f);
		System.out.println("img col " + MatrixUtil.pretty(r));
		conv.setPre(new InputLayer("i", 0, 0).setInput(f));
		FloatMatrix z = conv.forward();
		System.out.println("dot z " + MatrixUtil.pretty(z));
		FloatMatrix delta = FloatMatrix.rand(8, 2);
		conv.setDelta(delta);
		FloatMatrix d = conv.backward();
		System.out.println("pre delta " + MatrixUtil.pretty(d));
	}

	@Test
	public void testConvPool() {
		Context.init();
		FloatMatrix f = FloatMatrix.zeros(28*28, 1);
		int k = 1;
		for (int i=0; i<f.rows; i++) {
			for (int j=0; j<f.columns; j++) {
				f.data[f.index(i, j)] = k++;
			}
		}
		System.out.println("origin " + MatrixUtil.pretty(f));
		Conv2DLayer conv = new Conv2DLayer("conv", 28, 28, 1, 3, 1, 1, 0);
		conv.setActivation(new Relu());
		conv.setPadding(0);
		System.out.println("conv output "+ conv.getOutputW() + " "+conv.getOutputH());
		PoolingLayer pool = new PoolingLayer("pool", conv.getOutputW(), conv.getOutputH(), conv.getK(), 2, 2);
		pool.setPadding(1);
		System.out.println("pool output "+ pool.getOutputW() + " "+pool.getOutputH());
		conv.setPre(new InputLayer("i", 0, 0).setInput(f));
		conv.setNext(pool);
		FloatMatrix z = conv.forward();
		System.out.println("dot z " + MatrixUtil.pretty(z));
		FloatMatrix p = pool.forward();
		System.out.println("pooled z " + MatrixUtil.pretty(p));
	}

	@Test
	public void testPool() {
		Context.init();
		FloatMatrix f = FloatMatrix.zeros(16, 2);
		int k = 1;
		for (int i=0; i<f.columns; i++) {
			for (int j=0; j<f.rows; j++) {
				f.data[f.index(j, i)] = k++;
			}
		}
		System.out.println("origin "+MatrixUtil.pretty(f));
		PoolingLayer pool = new PoolingLayer("pool", 4, 4, 1, 2, 2);
		pool.setPre(new InputLayer("i", 0, 0).setInput(f));
		FloatMatrix z = pool.forward();
		System.out.println("down sample z "+MatrixUtil.pretty(z));

		pool.setDelta(z);
		FloatMatrix delta = pool.backward();
		System.out.println("up sample delta "+MatrixUtil.pretty(delta));
	}

}

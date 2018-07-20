import context.Context;
import javafx.scene.effect.FloatMap;
import layer.Conv2DLayer;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.junit.Test;
import util.MatrixUtil;

public class TestConv {

	@Test
	public void testImg2Col() {
		Context.init();
		FloatMatrix f = FloatMatrix.zeros(9, 1);
		int k = 1;
		for (int i=0; i<f.rows; i++) {
			for (int j=0; j<f.columns; j++) {
				f.data[f.index(i, j)] = k++;
			}
		}
		System.out.println("origin "+MatrixUtil.pretty(f));
		Conv2DLayer conv =new Conv2DLayer("conv", 3, 3, 1, 2, 1, 2);
		conv.setPadding(1);
		FloatMatrix r = conv.img2Col(f);
		System.out.println("img col "+MatrixUtil.pretty(r));
		FloatMatrix z = conv.forward(f);
		System.out.println("dot z "+MatrixUtil.pretty(z));
	}
}

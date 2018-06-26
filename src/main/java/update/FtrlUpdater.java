package update;

import lombok.Data;
import org.apache.commons.lang.StringUtils;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.MatrixUtil;

import java.util.HashMap;
import java.util.Map;

/**
 * Ftrl: a Method for Stochastic Optimization.
 *
 * https://github.com/fmfn/FTRLp/blob/master/FTRLp.py
 * https://www.cnblogs.com/EE-NovRain/p/3810737.html
 */
@Data
public class FtrlUpdater implements Updater {

	private static Logger logger = LoggerFactory.getLogger(FtrlUpdater.class);

	private float alfa;
	private float beta;
	private float l1;
	private float l2;

	private Map<String, FloatMatrix> Z;
	private Map<String, FloatMatrix> N;

	public FtrlUpdater() {}

    public FtrlUpdater(float alfa, float beta, float l1, float l2) {
		this.alfa = alfa;
		this.beta = beta;
		this.l1 = l1;
		this.l2 = l2;
    }

	public FtrlUpdater(String str) {
		this.alfa = Float.parseFloat(StringUtils.substringBetween(str, "alfa:", "@"));
		this.beta = Float.parseFloat(StringUtils.substringBetween(str, "beta:", "@"));
		this.l1 = Float.parseFloat(StringUtils.substringBetween(str, "l1:", "@"));
		this.l2 = Float.parseFloat(StringUtils.substringBetween(str, "l2:", "@"));
	}

	public FloatMatrix update(String key, FloatMatrix w, FloatMatrix dw) {
		if (!N.containsKey(key)) {
			N.put(key, FloatMatrix.zeros(w.length));
		}
		if (!Z.containsKey(key)) {
			Z.put(key, FloatMatrix.zeros(w.length));
		}
		FloatMatrix s = MatrixFunctions.sqrt(N.get(key).add(MatrixFunctions.pow(dw, 2))).subi(MatrixFunctions.sqrt(N.get(key).div(this.alfa)));
		Z.put(key, Z.get(key).addi(dw.sub(s.mul(w))));
		N.put(key, N.get(key).addi(MatrixFunctions.pow(dw, 2)));

		FloatMatrix zi = Z.get(key);
		FloatMatrix ni = N.get(key);
		// update each param
		for (int i=0; i<w.data.length; i++) {
			if (Math.abs(zi.data[i]) <= l1) {
				w.data[i] = 0;
			} else {
				float sign = zi.data[i] >= 0 ? 1 : -1;
				w.data[i] = -(zi.data[i] - sign * l1) / ((l2 + (beta + (float)Math.sqrt(ni.data[i]))) / alfa);
			}
		}
		return w;
	}

	public String getName() {
		return "adam@alfa:"+alfa+"@beta:"+beta+"@l1:"+l1+"@l2:"+l2+"@";
	}
}

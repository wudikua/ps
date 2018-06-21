package update;

import lombok.Data;
import org.apache.commons.lang.StringUtils;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Adam: a Method for Stochastic Optimization.
 *
 * t <- t+1
 * Mt <- beta1 * M(t-1) + (1-beta1) *Gt
 * Vt <- beta2 * V(t-1) + (1-beta2) * Gt^2
 * Mt, <- Mt / (1 - beta1)
 * Vt, <- Vt / (1 - beta2)
 *
 * Wt <- W(t-1) - alfa * Mt, / (sqrt(Vt,) + epsilon)
 *
 * https://arxiv.org/abs/1412.6980
 * https://cloud.tencent.com/developer/article/1057062
 *
 */
@Data
public class AdamUpdater implements Updater {

	private static Logger logger = LoggerFactory.getLogger(AdamUpdater.class);

	private float alfa;
	private float beta1;
	private float beta2;
	private float epsilon;

	private Map<String, FloatMatrix> M;
	private Map<String, FloatMatrix> V;
	
	public AdamUpdater() {}

    public AdamUpdater(double alfa, double beta1, double beta2, double epsilon) {
		this.alfa = (float)alfa;
		this.beta1 = (float)beta1;
		this.beta2 = (float)beta2;
		this.epsilon = (float)epsilon;
    }

	public AdamUpdater(String str) {
		this.alfa = Float.parseFloat(StringUtils.substringBetween(str, "alfa:", "@"));
		this.beta1 = Float.parseFloat(StringUtils.substringBetween(str, "beta1:", "@"));
		this.beta2 = Float.parseFloat(StringUtils.substringBetween(str, "beta2:", "@"));
		this.epsilon = Float.parseFloat(StringUtils.substringBetween(str, "epsilon:", "@"));
	}

	public FloatMatrix update(String key, FloatMatrix w, FloatMatrix dw) {
		if (M == null || V == null || !M.containsKey(key) || !M.containsKey(key)) {
			initMandV(key, dw);
		}
		M.put(key, dw.mul(1 - beta1).addi(M.get(key).muli(beta1)));
		V.put(key, dw.mul(dw).muli(1 - beta2).addi(V.get(key).muli(beta2)));
		FloatMatrix Mm = M.get(key).div(1 - beta1);
		FloatMatrix Vv = V.get(key).div(1 - beta2);
		if (w == null) {
			System.out.println(key);
			System.exit(0);
		}
		return w.addi(Mm.div(MatrixFunctions.sqrt(Vv).addi(epsilon)).muli(-1 * alfa));
	}

	public String getName() {
		return "adam@alfa:"+alfa+"@beta1:"+beta1+"@beta2:"+beta2+"@epsilon:"+epsilon+"@";
	}

	public void initMandV(String key, FloatMatrix dw) {
		FloatMatrix t = new FloatMatrix(dw.rows, dw.columns);
		if (M == null || V == null) {
			M = new HashMap<String, FloatMatrix>();
			V = new HashMap<String, FloatMatrix>();
		}
		M.put(key, t);
		V.put(key, t);
	}

}

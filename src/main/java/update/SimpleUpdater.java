package update;

import lombok.Data;
import org.apache.commons.lang.StringUtils;
import org.jblas.FloatMatrix;

@Data
public class SimpleUpdater implements Updater {

	private float eta;

	public SimpleUpdater(float eta) {
		this.eta = eta;
	}

	public SimpleUpdater(String str) {
		eta = Float.parseFloat(StringUtils.substringBetween(str, "eta:", "@"));
	}

	public FloatMatrix update(String t, FloatMatrix w, FloatMatrix dw) {
		return w.addi(dw.muli(-eta));
	}

	public String getName() {
		return "simple@eta:"+eta+"@";
	}
}

package update;


import org.jblas.FloatMatrix;

public interface Updater {

	public FloatMatrix update(String t, FloatMatrix w, FloatMatrix dw);

	public String getName();
}

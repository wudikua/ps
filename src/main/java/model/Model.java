package model;


import org.jblas.FloatMatrix;
import update.Updater;

import java.util.Map;

public interface Model {

	public float train(Map<String, FloatMatrix> datas);

	public FloatMatrix predict(Map<String, FloatMatrix> datas);

	public void update();

	public Map<String, Updater> getUpdater();
}

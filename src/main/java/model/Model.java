package model;


import org.jblas.FloatMatrix;
import update.Updater;

public interface Model {

	public void train(FloatMatrix E, FloatMatrix X, FloatMatrix Y);

	public FloatMatrix predict(FloatMatrix E, FloatMatrix X);

	public void update();

	public Updater getUpdater();
}

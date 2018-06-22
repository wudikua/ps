package store;

import org.jblas.FloatMatrix;

import java.util.concurrent.Callable;

public class MyKey {
	private String key;
	private Callable<FloatMatrix> init;

	public MyKey(String key, Callable<FloatMatrix> init) {
		this.key = key;
		this.init = init;
	}

	public String getKey() {
		return key;
	}

	public Callable<FloatMatrix> getValue() {
		return init;
	}

	@Override
	public int hashCode() {
		return this.key.hashCode();
	}

	@Override
	public boolean equals(Object obj) {
		MyKey o = (MyKey) obj;
		return this.getKey().equals(o.getKey());
	}
}

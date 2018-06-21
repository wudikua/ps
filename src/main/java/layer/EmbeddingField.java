package layer;

import java.util.Map;
import java.util.concurrent.Callable;

import org.jblas.FloatMatrix;
import org.jblas.JavaBlas;

import com.google.common.collect.Maps;

import activations.Activation;
import lombok.Data;
import store.KVStore;
import util.MatrixUtil;

@Data
public class EmbeddingField {
	protected String name;
	protected int inputDims, outputDims;
	protected Map<String, FloatMatrix> weights = Maps.newHashMap();
	protected Map<String, FloatMatrix> weightsGradient = Maps.newHashMap();
	protected Map<String, Integer> weightsGradientN = Maps.newHashMap();
	protected FloatMatrix Z;
	protected FloatMatrix A;
	protected Activation activation;
	protected float[] nSample;
	protected KVStore kvStore;
	protected Callable<FloatMatrix> initW;

	public EmbeddingField(String name, final int inputDims, final int outputDims) {
		this.name = name;
		this.inputDims = inputDims;
		this.outputDims = outputDims;
		this.kvStore = KVStore.ins();
		final float xavier = (float) (4 * (Math.sqrt(6) / Math.sqrt(inputDims + outputDims)));
		this.initW = new Callable<FloatMatrix>() {
			@Override
			public FloatMatrix call() throws Exception {
				return MatrixUtil.rand(outputDims, inputDims, xavier);
			}
		};
	}

	private void checkExists(String key) {
		if (!weights.containsKey(key)) {
			FloatMatrix weight = kvStore.get(key, initW);
			weights.put(key, weight);
		}
	}

	public FloatMatrix forward(float[] nSample) {
		this.nSample = nSample;
		FloatMatrix WX = new FloatMatrix(outputDims, nSample.length);
		for (int i =0; i<nSample.length; i++) {
			float sample = nSample[i];
			String key = this.name + "." + String.valueOf(sample);
			checkExists(key);
			JavaBlas.rcopy(weights.get(key).length, weights.get(key).data, 0, 1, WX.data, i*outputDims, 1);
		}
		Z = WX;
		A = activation.forward(Z);
		return A;
	}

	public void clear() {
		weights = Maps.newHashMap();
		weightsGradient.clear();
		weightsGradientN.clear();
	}

    public void backward(int offset, FloatMatrix delta) {
        int k = 0;
        for (double sample : nSample) {
            String key = this.name + "." + String.valueOf(sample);
            if (!weightsGradient.containsKey(key)) {
                weightsGradient.put(key, activation.backward(delta.getRange(offset, offset + outputDims, k, k + 1), Z, A.getColumn(k)));
                weightsGradientN.put(key, 1);
            } else {
                weightsGradient.put(key, weightsGradient.get(key).addi(activation.backward(delta.getRange(offset, offset + outputDims, k, k + 1), Z, A.getColumn(k))));
                weightsGradientN.put(key, weightsGradientN.get(key) + 1);
            }
            k++;
        }
        for (String key : weightsGradientN.keySet()) {
            weightsGradient.put(key, weightsGradient.get(key).divi(weightsGradientN.get(key).intValue()));
			kvStore.sum(key, weightsGradient.get(key));
        }

	}
}

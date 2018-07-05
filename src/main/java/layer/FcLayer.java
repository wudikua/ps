package layer;

import activations.Relu;
import activations.Sigmoid;
import com.google.common.collect.Lists;
import context.Context;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import activations.Activation;
import lombok.Data;
import store.KVStore;
import update.Updater;
import util.MatrixUtil;
import visual.UiClient;

import java.util.List;
import java.util.concurrent.Callable;

@Data
public class FcLayer extends Layer {
	private static Logger logger = LoggerFactory.getLogger(FcLayer.class);

	protected FloatMatrix weights;
	protected FloatMatrix weightsGradient;
	protected FloatMatrix bias;
	protected FloatMatrix biasGradient;
	protected FloatMatrix Z;
	protected Activation activation;
	protected Callable<FloatMatrix> initW;
	protected Callable<FloatMatrix> initB;

	public FcLayer(String name, final int inputDims, final int outputDims) {
		super(name, inputDims, outputDims);
		this.initW = new Callable<FloatMatrix>() {
			@Override
			public FloatMatrix call() throws Exception {
				final float xavier = (float) (4 * (Math.sqrt(6) / Math.sqrt(inputDims + outputDims)));
				return MatrixUtil.rand(outputDims, inputDims, xavier);
			}
		};
		this.initB = new Callable<FloatMatrix>() {
			@Override
			public FloatMatrix call() throws Exception {
				final float xavier = (float) (4 * (Math.sqrt(6) / Math.sqrt(inputDims + 1)));
				return MatrixUtil.rand(outputDims, 1, xavier);
			}
		};

		weights = this.kvStore.get(this.name + ".weights", initW);
		bias = this.kvStore.get(this.name + ".bias", initB);
	}

	// 构建多层全链接层，最后一层默认使用sigmoid
	public static List<Layer> build(int inputSize, int[] fcLayers) {
		List<Layer> result = Lists.newArrayList();
		for (int i=0; i<fcLayers.length; i++) {
			int outputSize = fcLayers[i];
			FcLayer fcLayer = new FcLayer("fc" + i, inputSize, outputSize);
			if (i == fcLayers.length - 1) {
				fcLayer.setActivation(new Sigmoid());
			} else {
				fcLayer.setActivation(new Relu());
			}
			result.add(fcLayer);
			if (i != 0) {
				result.get(i - 1).setNext(result.get(i));
			}
			inputSize = outputSize;
		}
		return result;
	}

	public void clear() {}

	public FloatMatrix forward() {
		FloatMatrix A = this.pre.A;
		FloatMatrix WX = weights.mmul(A);
		Z = WX.addiColumnVector(bias);
		if (logger.isDebugEnabled()) {
			logger.debug("Z {}", MatrixUtil.pretty(Z));
		}
		// 没有激活函数
		if (activation != null) {
			this.A = activation.forward(Z);
		} else {
			this.A = Z;
		}
		if (Context.isTraining() && Context.isReportUi()) {
			UiClient.ins().plot(name + ".output.mean", this.A.mean(), Context.step.get());
		}
		return this.A;
	}

	public FloatMatrix backward() {
		FloatMatrix delta;
		if (next == null) {
			delta = this.delta;
		} else {
			delta = next.delta;
		}
		if (activation != null) {
			delta = activation.backward(delta, Z, A);
		}
		biasGradient = delta.rowMeans();
		kvStore.sum(name+".bias", biasGradient);
		weightsGradient = (delta.mmul(pre.getA().transpose())).divi(delta.columns);
		kvStore.sum(name+".weights", weightsGradient);
		// update delta
		this.delta = weights.transpose().mmul(delta);
		return this.delta;
	}

	public void pullWeights() {
		weights = this.kvStore.get(this.name + ".weights", initW);
		bias = this.kvStore.get(this.name + ".bias", initB);
	}

}

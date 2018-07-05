package layer;

import activations.Activation;
import activations.Relu;
import activations.Sigmoid;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import context.Context;
import lombok.Data;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.MatrixUtil;
import visual.UiClient;

import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

@Data
public class LRLayer extends Layer {
	private static Logger logger = LoggerFactory.getLogger(LRLayer.class);

	// key是稀疏下标i，value wi是1X1的矩阵
	protected Map<String, FloatMatrix> weights = Maps.newHashMap();
	protected Map<String, FloatMatrix> weightsGradient = Maps.newHashMap();
	protected Map<String, Integer> weightsGradientN = Maps.newHashMap();
	// bias是1X1的矩阵
	protected FloatMatrix bias;
	protected FloatMatrix biasGradient;
	protected FloatMatrix Z;
	protected Activation activation;
	protected Callable<FloatMatrix> initW;
	protected Callable<FloatMatrix> initB;

	public LRLayer(String name, final int inputDims) {
		super(name, inputDims, 1);
		this.initW = new Callable<FloatMatrix>() {
			@Override
			public FloatMatrix call() throws Exception {
				return FloatMatrix.zeros(1);
			}
		};
		this.initB = new Callable<FloatMatrix>() {
			@Override
			public FloatMatrix call() throws Exception {
				return FloatMatrix.zeros(1);
			}
		};

		// weights 按需初始化
		bias = kvStore.get(this.name + ".bias", initB);
	}

	public void clear() {
		weights.clear();
		weightsGradient.clear();
		weightsGradientN.clear();
	}

	public FloatMatrix forward() {
		FloatMatrix X = this.pre.A;
		// 分布式worker 预处理权重
		if (Context.isDistributed() && !Context.isPServer()) {
			for (int i=0; i<X.data.length; i++) {
				kvStore.asyncGet(this.name + ".weights." + X.data[i], initW);
			}
			kvStore.asyncWait();
		}
		FloatMatrix WX = new FloatMatrix(1, X.columns);
		// 遍历每个sample
		for (int i=0; i<X.columns; i++) {
			float sumW = 0f;
			// dot特征
			for (int j=0; j<X.rows; j++) {
				float index = X.data[X.index(j, i)];
				FloatMatrix wi = kvStore.get(this.name + ".weights." + index, initW);
				weights.put(this.name + ".weights." + index, wi);
				sumW += wi.get(0);
			}
			WX.put(0, i, sumW);
		}
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
		delta = delta.rowMeans();
		// db = delta
		biasGradient = delta;
		kvStore.sum(name+".bias", biasGradient);
		// dw = delta * xi = delta * 1
		for (String key : weights.keySet()) {
			kvStore.sum(key, delta);
		}

		return this.delta;
	}

	public void pullWeights() {
		bias = this.kvStore.get(this.name + ".bias", initB);
	}

}

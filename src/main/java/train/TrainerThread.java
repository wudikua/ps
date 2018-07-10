package train;

import context.Context;
import model.Model;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.Callable;

public class TrainerThread implements Callable<Float> {

	static Logger logger = LoggerFactory.getLogger(TrainerThread.class);

	Map<String, FloatMatrix> datas;

	Model nn;

	int modelIndex;

	public TrainerThread(int modelIndex , Model nn, Map<String, FloatMatrix> datas) {
		this.modelIndex = modelIndex;
		this.datas = datas;
		this.nn = nn;
	}

	@Override
	public Float call() throws Exception {
		try {
			Context.modelIndex.set(modelIndex);
			// 拉取权重
			nn.pullWeights();
			return nn.train(datas);
		} catch (Exception e) {
			logger.error("trainer error", e);
		}
		return 0f;
	}
}

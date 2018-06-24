package train;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import data.TestDataSet;
import javafx.scene.effect.FloatMap;
import lombok.Data;
import model.DNN;
import model.Model;
import org.apache.commons.lang3.tuple.Pair;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import store.KVStore;

import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

@Data
public class Trainer {

	private static Logger logger = LoggerFactory.getLogger(Trainer.class);

	private int thread;

	private ExecutorService service;

	private List<Model> models;

	private static class TrainerThread implements Callable<Void> {
		Map<String, FloatMatrix> datas;
		Model nn;
		public TrainerThread(Model nn, Map<String, FloatMatrix> datas) {
			this.datas = datas;
			this.nn = nn;
		}

		@Override
		public Void call() throws Exception {
			try {
				nn.train(datas);
			} catch (Exception e) {
				logger.error("trainer error", e);
			}
			return null;
		}
	}

	public Trainer(int nThreads, Callable<Model> modelCallable) throws Exception {
		this.thread = nThreads;
		this.service = Executors.newFixedThreadPool(thread);
		this.models = initModels(modelCallable);
	}

	private List<Model> initModels(Callable<Model> modelCallable) throws Exception {
		List<Model> result = Lists.newArrayList();
		int n = thread + 1;
		for (int i=0; i< n; i++) {
			Model nn = modelCallable.call();
			result.add(nn);
		}
		return result;
	}

	public void run(List<Map<String, FloatMatrix>> dataList) {
		List<Future<Void>> futures = Lists.newArrayList();
		// multi thread submit
		for (int i=1; i< models.size() && i <= dataList.size(); i++) {
			Map<String, FloatMatrix> datas = Maps.newHashMap();
			Future<Void> future = service.submit(new TrainerThread(models.get(i), dataList.get(i-1)));
			futures.add(future);
		}
		// wait for every thread finish
		for (Future f : futures) {
			try {
				f.get(1000, TimeUnit.SECONDS);
			} catch (InterruptedException | ExecutionException | TimeoutException e) {
				logger.error("train error", e);
			}
		}
		// avg every thread weights
		Model model = models.get(0);
		logger.info("update model params");
		KVStore.ins().update(model.getUpdater());
		// update weights
		for (int i=1; i< models.size(); i++) {
			// 清理model中的权重
			models.get(i).update();
		}
		// 清理缓存中的权重
		KVStore.ins().clear();
	}

	public Model getTrainResult() {
		return models.get(0);
	}
}

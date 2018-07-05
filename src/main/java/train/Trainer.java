package train;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import context.Context;
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
import visual.UiClient;
import visual.UiServer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

@Data
public class Trainer {

	private static Logger logger = LoggerFactory.getLogger(Trainer.class);

	private int thread;

	private ExecutorService service;

	private List<Model> models;

	private static class TrainerThread implements Callable<Float> {
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
				return nn.train(datas);
			} catch (Exception e) {
				logger.error("trainer error", e);
			}
			return 0f;
		}
	}

	private static class PredictThread implements Callable<FloatMatrix> {

		Map<String, FloatMatrix> datas;
		Model nn;
		int modelIndex;

		public PredictThread(int modelIndex, Model nn, Map<String, FloatMatrix> datas) {
			this.modelIndex = modelIndex;
			this.datas = datas;
			this.nn = nn;
		}

		@Override
		public FloatMatrix call() throws Exception {
			try {
				Context.modelIndex.set(modelIndex);
				return nn.predict(datas);
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
		int n = thread;
		for (int i=0; i< n; i++) {
			Model nn = modelCallable.call();
			result.add(nn);
		}
		return result;
	}

	public FloatMatrix[] predict(List<Map<String, FloatMatrix>> dataList) {
		Context.status = Context.Stat.PREDICTING;
		List<Future<FloatMatrix>> futures = Lists.newArrayList();
		if (dataList.size() > models.size()) {
			throw new RuntimeException("dataList size > thread size");
		}
		// multi thread submit
		for (int i=0; i< models.size() && i < dataList.size(); i++) {
			Future<FloatMatrix> future = service.submit(new PredictThread(i, models.get(i), dataList.get(i)));
			futures.add(future);
		}
		FloatMatrix[] y = new FloatMatrix[dataList.size()];
		// wait for every thread finish
		for (int i=0; i<futures.size(); i++) {
			try {
				Future<FloatMatrix> f = futures.get(i);
				y[i] = f.get(1000, TimeUnit.SECONDS);
			} catch (InterruptedException | ExecutionException | TimeoutException e) {
				logger.error("train error", e);
			}
		}
		// 清理缓存中的权重
		KVStore.ins().clear();
		return y;
	}

	public void train(List<Map<String, FloatMatrix>> dataList) {
		Context.status = Context.Stat.TRAINING;
		List<Future<Float>> futures = Lists.newArrayList();
		if (dataList.size() > models.size()) {
			throw new RuntimeException("dataList size > thread size");
		}
		// multi thread submit
		for (int i=0; i< models.size() && i < dataList.size(); i++) {
			Future<Float> future = service.submit(new TrainerThread(i, models.get(i), dataList.get(i)));
			futures.add(future);
		}
		float loss =0;
		// wait for every thread finish
		for (Future<Float> f : futures) {
			try {
				loss += f.get(1000, TimeUnit.SECONDS);
			} catch (InterruptedException | ExecutionException | TimeoutException e) {
				logger.error("train error", e);
			}
		}
		// avg every thread weights
		Model model = models.get(0);
		logger.info("update model params");
		KVStore.ins().update(model.getUpdater());
		// update weights
		for (int i=0; i< models.size(); i++) {
			// 清理model中的权重
			models.get(i).update();
		}
		// 清理缓存中的权重
		KVStore.ins().clear();
		if (loss != 0) {
			// report meta
			long step = Context.step.incrementAndGet();
			UiClient.ins().plot("loss", loss / Context.thread, step);
		}
	}

	public Model getTrainResult() {
		return models.get(0);
	}
}

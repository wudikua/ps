
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import data.*;
import evaluate.AUC;
import evaluate.LossSurface;
import loss.CrossEntropy;
import model.DNN;
import model.Model;
import net.PServer;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import context.Context;
import train.Trainer;
import update.AdamUpdater;
import update.FtrlUpdater;
import update.Updater;
import util.MatrixUtil;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

public class CTR extends DataSet {

	private static Logger logger = LoggerFactory.getLogger(CTR.class);

	public static final int wideSize = 100000;

	static DataSet trainSet;

	static DataSet testSet;

	public CTR(Parser parser, DataSource source, int batch, int thread) {
		super(parser, source, batch, thread);
	}

	@Override
	public Map<String, FloatMatrix> parseFeature(List<List<Feature>> dataList) {
		Map<String, FloatMatrix> map = Maps.newHashMap();
		int N = dataList.size();
		float[][] E = new float[23][N];
		float[][] X = new float[45][N];
		float[][] Y = new float[1][N];
		for (int i=0; i<dataList.size(); i++) {
			List<Feature> cols = dataList.get(i);
			Y[0][i] = cols.get(0).toF();
			for (int j=1; j<24; j++) {
				E[j-1][i] = cols.get(j).getIdx();
			}
			for (int j=24; j<69; j++) {
				X[j-24][i] = cols.get(j).toF();
			}
		}
		map.put("X", new FloatMatrix(X));
		map.put("E", new FloatMatrix(E));
		map.put("W", MatrixUtil.hash(map.get("E"), wideSize));
		map.put("Y", new FloatMatrix(Y));
		return map;
	}

	public static void main(String args[]) throws Exception {
		Context.init();
		Context.thread = 1;
		if (Context.isPServer()) {
			// 启动PS进程
			Updater updater = new AdamUpdater(0.005, 0.9, 0.999, Math.pow(10, -8));
			Updater ftrl = new FtrlUpdater(0.005f, 1f, 0.001f, 0.001f);
			PServer server = new PServer(Context.psPort, Context.workerNum);
			server.getUpdaterMap().put(updater.getName(), updater);
			server.getUpdaterMap().put(ftrl.getName(), ftrl);
			server.start();
			System.exit(0);
		}
		trainSet = new CTR(new LibsvmParser(), new FileSource(new File(System.getProperty("train",
				CTR.class.getResource("").getPath()+"../../src/main/resources/train.txt"))), 1000, 1);

		testSet = new CTR(new LibsvmParser(), new FileSource(new File(System.getProperty("test",
				CTR.class.getResource("").getPath()+"../../src/main/resources/test.txt"))), 100, 1);
		Trainer trainer = new Trainer(Context.thread, new Callable<Model>() {
			@Override
			public Model call() throws Exception {
				return DNN.buildModel(23, 10, 45, new int[]{150, 10, 1});
			}
		});
		for (int epoch = 0; epoch < 100 && !Context.finish; epoch++) {
			logger.info("epoch {}", epoch);
			train(trainer);

			auc(trainer);

			loss_surface(trainer);
		}
	}

	private static void loss_surface(Trainer trainer) throws IOException {
		logger.info("compute loss surface");
		Map<String, FloatMatrix> datas = testSet.next();
		LossSurface lossSurface = new LossSurface(datas, datas.get("Y"), new CrossEntropy(), trainer.getTrainResult());
		lossSurface.plot();
		logger.info("loss surface draw finish");
		testSet.reset();
	}

	private static void train(Trainer trainer) throws IOException {
		while (trainSet.hasNext()) {
			List<Map<String, FloatMatrix>> dataList = Lists.newArrayList();
			for (int i=0; i<Context.thread; i++) {
				Map<String, FloatMatrix> datas = trainSet.next();
				if (datas == null || datas.isEmpty()) {
					logger.info("data read eof");
					break;
				}
				dataList.add(datas);
			}
			trainer.train(dataList);
		}
		trainSet.reset();
	}

	private static void auc(Trainer trainer) throws IOException {
		logger.info("begin compute auc...");
		List<Pair<Double, Double>> data = new ArrayList<Pair<Double, Double>>();
		while (testSet.hasNext()) {
			List<Map<String, FloatMatrix>> dataList = Lists.newArrayList();
			List<FloatMatrix> y = Lists.newArrayList();
			for (int i=0; i<Context.thread; i++) {
				Map<String, FloatMatrix> datas = testSet.next();
				if (datas == null || datas.isEmpty()) {
					logger.info("read data eof");
					break;
				}
				y.add(datas.get("Y"));
				dataList.add(datas);
			}
			FloatMatrix[] ps = trainer.predict(dataList);
			for (int i=0; i<y.size(); i++) {
				if (ps[i] == null) {
					continue;
				}
				for (int j=0; j<y.get(i).length; j++) {
					data.add(new MutablePair<Double, Double>((double) ps[i].data[j], (double) y.get(i).data[j]));
				}
			}
		}
		AUC auc = new AUC(data);
		logger.info("AUC {}", auc.calculate());
		testSet.reset();
	}
}

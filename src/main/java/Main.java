
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import data.TestDataSet;
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

public class Main {

	private static Logger logger = LoggerFactory.getLogger(Main.class);

	public static final int wideSize = 100000;

	static BufferedReader train;

	static BufferedReader test;

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
		train = new BufferedReader(new FileReader(new File(System.getProperty("train",
				Main.class.getResource("").getPath()+"../../src/main/resources/train.txt"))));
		test = new BufferedReader(new FileReader(new File(System.getProperty("test",
				Main.class.getResource("").getPath()+"../../src/main/resources/test.txt"))));
		Trainer trainer = new Trainer(Context.thread, new Callable<Model>() {
			@Override
			public Model call() throws Exception {
				return DNN.buildModel(23, 10, 45, new int[]{150, 10, 1});
//				return WideDeepNN.buildModel(23, 10, 45, new int[]{1000, 100, 1}, wideSize);
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
		Pair<TestDataSet.MatrixData, Boolean> d = TestDataSet.fromStream(test, Integer.parseInt(System.getProperty("batch", "100")));
		Map<String, FloatMatrix> datas = Maps.newHashMap();
		datas.put("E", d.getKey().getE());
		datas.put("X", d.getKey().getX());
		datas.put("W", MatrixUtil.hash(d.getKey().getE(), wideSize));
		LossSurface lossSurface = new LossSurface(datas, d.getKey().getY(), new CrossEntropy(), trainer.getTrainResult());
		lossSurface.plot();
		logger.info("loss surface draw finish");
		test.close();
		test = new BufferedReader(new FileReader(new File(System.getProperty("test",
				Main.class.getResource("").getPath()+"../../src/main/resources/test.txt"))));
	}

	private static void train(Trainer trainer) throws IOException {
		boolean eof = false;
		while (!Context.finish && !eof) {
			List<Map<String, FloatMatrix>> dataList = Lists.newArrayList();
			for (int i=0; i<Context.thread; i++) {
				Pair<TestDataSet.MatrixData, Boolean> d = TestDataSet.fromStream(train, Integer.parseInt(System.getProperty("batch", "10000")));
				if (!d.getValue()) {
					logger.info("data read eof");
					eof = true;
					break;
				}
				Map<String, FloatMatrix> datas = Maps.newHashMap();
				datas.put("E", d.getKey().getE());
				datas.put("X", d.getKey().getX());
				datas.put("W", MatrixUtil.hash(d.getKey().getE(), wideSize));
				datas.put("Y", d.getKey().getY());
				dataList.add(datas);
			}
			trainer.train(dataList);
		}
		train.close();
		train = new BufferedReader(new FileReader(new File(System.getProperty("train",
				Main.class.getResource("").getPath()+"../../src/main/resources/train.txt"))));
	}

	private static void auc(Trainer trainer) throws IOException {
		logger.info("begin compute auc...");
		List<Pair<Double, Double>> data = new ArrayList<Pair<Double, Double>>();
		boolean breakPredict = false;
		while (!breakPredict) {
			List<Map<String, FloatMatrix>> dataList = Lists.newArrayList();
			List<FloatMatrix> y = Lists.newArrayList();
			for (int i=0; i<Context.thread; i++) {
				Pair<TestDataSet.MatrixData, Boolean> d = TestDataSet.fromStream(test, Integer.parseInt(System.getProperty("batch", "1000")));
				if (!d.getValue()) {
					logger.info("data read eof");
					breakPredict = true;
					break;
				}
				Map<String, FloatMatrix> datas = Maps.newHashMap();
				datas.put("E", d.getKey().getE());
				datas.put("X", d.getKey().getX());
				datas.put("W", MatrixUtil.hash(d.getKey().getE(), wideSize));
				y.add(d.getKey().getY());
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
		test.close();
		test = new BufferedReader(new FileReader(new File(System.getProperty("test",
				Main.class.getResource("").getPath()+"../../src/main/resources/test.txt"))));
	}
}

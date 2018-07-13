import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import context.Context;
import data.TestDataSet;
import evaluate.AUC;
import evaluate.LossSurface;
import evaluate.SoftmaxPrecision;
import loss.CrossEntropy;
import model.DNN;
import model.FullConnectedNN;
import model.Model;
import net.PServer;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
import java.util.concurrent.Callable;

public class Mnist {

	private static Logger logger = LoggerFactory.getLogger(CTR.class);

	static BufferedReader train;

	static BufferedReader test;

	public static void main(String args[]) throws Exception {
		Context.init();
		Context.thread = 4;
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
				CTR.class.getResource("").getPath()+"../../src/main/resources/mnist_train.csv"))));
		test = new BufferedReader(new FileReader(new File(System.getProperty("test",
				CTR.class.getResource("").getPath()+"../../src/main/resources/mnist_test.csv"))));
		Trainer trainer = new Trainer(Context.thread, new Callable<Model>() {
			@Override
			public Model call() throws Exception {
				return FullConnectedNN.buildModel(784, new int[]{150, 50, 10});
			}
		});
		for (int epoch = 0; epoch < 100 && !Context.finish; epoch++) {
			logger.info("epoch {}", epoch);
			train(trainer);

			precision(trainer);
		}
	}

	private static void train(Trainer trainer) throws IOException {
		boolean eof = false;
		while (!Context.finish && !eof) {
			List<Map<String, FloatMatrix>> dataList = Lists.newArrayList();
			for (int i=0; i<Context.thread; i++) {
				Pair<TestDataSet.MatrixData, Boolean> d = TestDataSet.fromMnistStream(train, Integer.parseInt(System.getProperty("batch", "1000")));
				if (!d.getValue()) {
					logger.info("data read eof");
					eof = true;
					break;
				}
				Map<String, FloatMatrix> datas = Maps.newHashMap();
				datas.put("X", d.getKey().getX());
				datas.put("Y", d.getKey().getY());
				dataList.add(datas);
			}
			trainer.train(dataList);
		}
		train.close();
		train = new BufferedReader(new FileReader(new File(System.getProperty("train",
				CTR.class.getResource("").getPath()+"../../src/main/resources/mnist_train.csv"))));
	}

	private static void precision(Trainer trainer) throws IOException {
		logger.info("begin compute precision...");
		List<Float[]> p = Lists.newArrayList();
		List<Float> l = Lists.newArrayList();
		boolean breakPredict = false;
		while (!breakPredict) {
			List<Map<String, FloatMatrix>> dataList = Lists.newArrayList();
			List<FloatMatrix> y = Lists.newArrayList();
			for (int i=0; i<Context.thread; i++) {
				Pair<TestDataSet.MatrixData, Boolean> d = TestDataSet.fromMnistStream(test, Integer.parseInt(System.getProperty("batch", "1000")));
				if (!d.getValue()) {
					logger.info("data read eof");
					breakPredict = true;
					break;
				}
				Map<String, FloatMatrix> datas = Maps.newHashMap();
				datas.put("X", d.getKey().getX());
				y.add(d.getKey().getY());
				dataList.add(datas);
			}
			FloatMatrix[] ps = trainer.predict(dataList);
			for (int i=0; i<y.size(); i++) {
				if (ps[i] == null) {
					continue;
				}
				for (int j=0; j<y.get(i).length; j++) {
					l.add(y.get(i).data[j]);
					Float[] c = new Float[ps[i].rows];
					for (int k=0; k<ps[i].rows; k++) {
						c[k] = ps[i].data[ps[i].index(k, j)];
					}
					p.add(c);
				}
			}
		}
		SoftmaxPrecision precision = new SoftmaxPrecision(l, p);
		logger.info("Precision {}", precision.calculate());
		test.close();
		test = new BufferedReader(new FileReader(new File(System.getProperty("test",
				CTR.class.getResource("").getPath()+"../../src/main/resources/mnist_test.csv"))));
	}
}

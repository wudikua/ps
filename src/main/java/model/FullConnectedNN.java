package model;

import activations.Softmax;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import context.Context;
import evaluate.AUC;
import evaluate.SoftmaxPrecision;
import layer.*;
import lombok.Data;
import loss.CrossEntropy;
import loss.Loss;
import loss.SoftmaxLoss;
import org.apache.commons.math3.util.Precision;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import update.AdamUpdater;
import update.Updater;

import java.util.List;
import java.util.Map;

@Data
public class FullConnectedNN implements Model {

	private static Logger logger = LoggerFactory.getLogger(FullConnectedNN.class);

	private Loss loss = new SoftmaxLoss();

	private List<Layer> layers = Lists.newArrayList();

	private List<Layer> inputs = Lists.newArrayList();

	private Map<String, Updater> updater = Maps.newHashMap();

	public float train(Map<String, FloatMatrix> datas) {
		FloatMatrix X = datas.get("X");
		FloatMatrix Y = datas.get("Y");
		// number
		inputs.get(0).setA(X);
		// 前向
		for (Layer layer : layers) {
			layer.forward();
		}
		// loss计算
		FloatMatrix P = layers.get(layers.size()-1).getA();
		float lossVal = loss.forward(P, Y);
		FloatMatrix delta = loss.backward(P , Y);
		logger.debug("\nP {}\nY {}\ndelta {}", P, Y, delta);
		logger.info("LOSS is {}", lossVal);
		if (Context.step.get() % Context.nTermDump == 0) {
			SoftmaxPrecision precision = new SoftmaxPrecision(Y, P);
			logger.info("Train Precision {}", precision.calculate());
		}
		if (lossVal <= CrossEntropy.slim || Float.isNaN(lossVal)) {
			Context.finish = true;
			logger.info("Oh Yeah lossVal is too slim !!! model train success");
			return lossVal;
		}
		layers.get(layers.size() - 1).setDelta(delta);
		// 反向求导
		for (int i=layers.size() - 1; i>=0; i--) {
			layers.get(i).backward();
		}
		return lossVal;
	}

	public void pullWeights() {
		for (int i=0; i<layers.size(); i++) {
			layers.get(i).pullWeights();
		}
	}

	public FloatMatrix predict(Map<String, FloatMatrix> datas) {
		FloatMatrix X = datas.get("X");
		// number
		inputs.get(0).setA(X);
		// 前向
		for (Layer layer : layers) {
			layer.forward();
		}
		return layers.get(layers.size()-1).getA();
	}

	public static FullConnectedNN buildModel(int numberFieldNum, int[] fcLayerDims) {
		// model construct
		FullConnectedNN nn = new FullConnectedNN();
		nn.getUpdater().put("default", new AdamUpdater(0.005, 0.9, 0.999, Math.pow(10, -8)));
		nn.setLoss(new SoftmaxLoss());
		List<Layer> layers = Lists.newArrayList();
		List<Layer> inputs = Lists.newArrayList();
		// 输入层
		Layer numberFeatureLayer = new InputLayer("number", 0, numberFieldNum).setIsInput(true);
		// 全连接
		List<Layer> fcLayers = FcLayer.build(numberFeatureLayer.getOutputDims(), fcLayerDims);
		((FcLayer)fcLayers.get(fcLayers.size() - 1)).setActivation(new Softmax());
		/**
		 * 组织层关系
		 * number -> full connected
		 */
		numberFeatureLayer.setNext(fcLayers.get(0));
		// 添加到layers
		layers.addAll(fcLayers);
		// 添加到inputs
		inputs.add(numberFeatureLayer);
		nn.setInputs(inputs);
		nn.setLayers(layers);
		return nn;
	}

}

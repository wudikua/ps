package layer;


import activations.Activation;
import com.google.common.collect.Maps;
import lombok.Data;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import store.KVStore;
import util.MatrixUtil;

import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

@Data
public class Conv2DLayer {

	String name;

	int inputW;

	int inputH;

	int inputD;

	// numbers of filters
	int K;

	int stride;

	int kernelSize;

	int padding = 0;

	// 将K个卷积核展开，行为K个卷积核，列为卷积核矩阵的行表达形式
	FloatMatrix weights;

	FloatMatrix bias;

	FloatMatrix A;

	FloatMatrix Z;

	Activation activation;

	protected Callable<FloatMatrix> initW;

	protected Callable<FloatMatrix> initB;

	protected KVStore kvStore = KVStore.ins();

	public Conv2DLayer(String name, int inputW, int inputH, int inputD, int kernelSize, int stride, int outputNum) {
		this.name = name;
		this.inputW = inputW;
		this.inputH = inputH;
		this.inputD = inputD;
		this.K = outputNum;
		this.stride = stride;
		this.kernelSize = kernelSize;

		this.initW = new Callable<FloatMatrix>() {
			@Override
			public FloatMatrix call() throws Exception {
				return FloatMatrix.rand(K, inputD * kernelSize * kernelSize);
			}
		};
		this.initB = new Callable<FloatMatrix>() {
			@Override
			public FloatMatrix call() throws Exception {
				return FloatMatrix.rand(1);
			}
		};
		weights = kvStore.get(name+".weighs", initW);
		bias = kvStore.get(name+".bias", initB);
	}

	private int getOutputW() {
		return ((inputW - kernelSize + 2*padding) / stride + 1);
	}

	private int getOutputH() {
		return ((inputH - kernelSize + 2*padding) / stride + 1);
	}

	private int getWeightsN() {
		return kernelSize * kernelSize * inputD * K;
	}

	private int getBiasN() {
		return K;
	}

	// https://buptldy.github.io/2016/10/01/2016-10-01-im2col/
	public FloatMatrix img2Col(FloatMatrix A) {
		int N = A.columns;
		int W = getOutputW();
		int H = getOutputH();
		float[][] img = new float[inputD * kernelSize * kernelSize][W * H * N];
		// 通道
		for (int d=0; d<inputD; d++) {
			for (int i=0; i<W; i++) {
				for (int j=0; j<H; j++) {
					for (int ki=0; ki<kernelSize; ki++) {
						for (int kj=0; kj<kernelSize; kj++) {
							// 前半部分是通道d的偏移，后半部分是通道内卷积核的偏移
							int row = d * kernelSize * kernelSize + ki * kernelSize + kj;
							for (int sample=0; sample<N; sample++) {
								// 第一部分行列的偏移，第二部分是多张图片的偏移
								int col = (j * H + i) + (sample * H * W);
								float pixel;
								int depthOffset = d * inputW * inputH;
								int rowOffset = stride * i + ki - padding;
								int colOffset = stride * j + kj - padding;
								if (rowOffset < 0 || colOffset < 0 || rowOffset >= inputW || colOffset >= inputH) {
									pixel = 0;
								} else {
									pixel = A.get(depthOffset + rowOffset * inputH + colOffset, sample);
								}
								img[row][col] = pixel;
							}
						}
					}
				}
			}
		}
		return new FloatMatrix(img);
	}

	public FloatMatrix forward(FloatMatrix A) {
		FloatMatrix imgVector = img2Col(A);
		Z = weights.mmul(imgVector);
		if (activation != null) {
			this.A = activation.forward(Z);
		} else {
			this.A = Z;
		}
		return this.A;
	}

	public FloatMatrix backward() {
		return null;
	}

	public void pullWeights() {
		weights = kvStore.get(name+".weighs", initW);
		bias = kvStore.get(name+".bias", initB);
	}
}

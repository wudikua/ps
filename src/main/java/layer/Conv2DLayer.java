package layer;


import activations.Activation;
import lombok.Data;
import org.jblas.FloatMatrix;
import store.KVStore;

import java.util.concurrent.Callable;

@Data
public class Conv2DLayer extends Layer {

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
	FloatMatrix weightsGradient;
	FloatMatrix bias;
	FloatMatrix biasGradient;

	FloatMatrix Z;

	FloatMatrix Col;

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
				return FloatMatrix.rand(K, 1);
			}
		};
		weights = kvStore.get(name+".weights", initW);
		bias = kvStore.get(name+".bias", initB);
		this.inputDims = inputD * inputW * inputD;
		this.outputDims = getOutputW() * getOutputH() * K;
	}

	public int getOutputW() {
		return ((inputW - kernelSize + 2*padding) / stride + 1);
	}

	public int getOutputH() {
		return ((inputH - kernelSize + 2*padding) / stride + 1);
	}

	// https://buptldy.github.io/2016/10/01/2016-10-01-im2col/
	// http://www.datakit.cn/blog/2016/03/23/bp_cnn.html
	public FloatMatrix im2col(FloatMatrix A) {
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
								int col = (i * W + j) + (sample * H * W);
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

	public FloatMatrix reshapeForward(FloatMatrix m, int N) {
		int W = getOutputW();
		int H = getOutputH();
		FloatMatrix result = FloatMatrix.zeros(K * W * H, N);
		for (int k=0; k<K; k++) {
			for (int i=0; i<W; i++) {
				for (int j=0; j<H; j++) {
					for (int sample=0; sample<N; sample++) {
						float f = m.get(k, (i * W + j) + (sample * H * W));
						result.put(k * W * H + i * W + j, sample, f);
					}
				}
			}
		}
		return result;
	}

	public FloatMatrix forward() {
		FloatMatrix A = this.pre.A;
		int N = A.columns;
		// d * w * h , N 做卷积转换为 c * kernel * kernel, w' * h' * N的矩阵
		Col = im2col(A);
		// 结果为 c, w' * h' * N
		FloatMatrix WX = weights.mmul(Col);
		Z = WX.addiColumnVector(bias);
		if (activation != null) {
			this.A = activation.forward(Z);
		} else {
			this.A = Z;
		}
		// c, w' * h' * N 转换为 c * w' * h', N的矩阵
		this.A = reshapeForward(this.A, N);
		return this.A;
	}

	public FloatMatrix reshapeBackward(FloatMatrix m, int N) {
		int W = getOutputW();
		int H = getOutputH();
		FloatMatrix result = FloatMatrix.zeros(K, W * H * N);
		for (int k=0; k<K; k++) {
			for (int i=0; i<W; i++) {
				for (int j=0; j<H; j++) {
					for (int sample=0; sample<N; sample++) {
						float f = m.get(k * W * H + i * W + j, sample);
						result.put(k, (i * W + j) + (sample * H * W), f);
					}
				}
			}
		}
		return result;
	}

	public FloatMatrix col2im(FloatMatrix A, int N) {
		int W = getOutputW();
		int H = getOutputH();
		float[][] img = new float[inputD * inputW * inputH][N];
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
								int col = (i * W + j) + (sample * H * W);
								float delta = A.get(row, col);
								int depthOffset = d * inputW * inputH;
								int rowOffset = stride * i + ki - padding;
								int colOffset = stride * j + kj - padding;
								img[depthOffset + rowOffset * inputH + colOffset][sample] += delta;
							}
						}
					}
				}
			}
		}
		return new FloatMatrix(img);
	}

	public FloatMatrix backward() {
		FloatMatrix delta;
		if (next == null) {
			delta = this.delta;
		} else {
			delta = next.delta;
		}
		int N = delta.columns;
		// c * w' * h', N 转换为 c, w' * h' * N的矩阵
		delta = reshapeBackward(delta, N);
		if (activation != null) {
			delta = activation.backward(delta, Z, A);
		}
		biasGradient = delta.rowMeans();
		kvStore.sum(name+".bias", biasGradient);
		weightsGradient = (delta.mmul(Col.transpose())).divi(delta.columns);
		kvStore.sum(name+".weights", weightsGradient);
		this.delta = weights.transpose().mmul(delta);
		// c, w' * h' * N 的矩阵反向传导转换为 d * w * h, N 的图片
		this.delta = col2im(this.delta, N);
		return this.delta;
	}

	public void pullWeights() {
		weights = kvStore.get(name+".weighs", initW);
		bias = kvStore.get(name+".bias", initB);
	}
}

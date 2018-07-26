package layer;


import activations.Activation;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multimaps;
import lombok.Data;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jblas.FloatMatrix;
import store.KVStore;

import java.util.List;
import java.util.concurrent.Callable;

@Data
public class PoolingLayer extends Layer {

	String name;

	int inputW;

	int inputH;

	int inputD;

	int K;

	int stride;

	int kernelSize;

	int padding = 0;

	// max pooling 时存储的位数组
	Multimap<Integer, Pair<Integer, Integer>> index;

	public PoolingLayer(String name, int inputW, int inputH, int inputD, int kernelSize, int stride) {
		this.name = name;
		this.inputW = inputW;
		this.inputH = inputH;
		this.inputD = inputD;
		this.K = inputD;
		this.stride = stride;
		this.kernelSize = kernelSize;
		this.index = ArrayListMultimap.create();
		this.inputDims = inputW * inputH * inputD;
		this.outputDims = getOutputW() * getOutputH() * inputD;
	}

	public int getOutputW() {
		return ((inputW - kernelSize + 2*padding) / stride + 1);
	}

	public int getOutputH() {
		return ((inputH - kernelSize + 2*padding) / stride + 1);
	}

	public FloatMatrix forward() {
		FloatMatrix A = this.pre.A;
		int N = A.columns;
		int W = getOutputW();
		int H = getOutputH();
		float[][] img = new float[inputD * W * H][N];
		// 通道
		for (int d=0; d<inputD; d++) {
			for (int i=0; i<W; i++) {
				for (int j=0; j<H; j++) {
					for (int sample = 0; sample < N; sample++) {
						float[] list = new float[kernelSize * kernelSize];
						int[] idx = new int[kernelSize * kernelSize];
						for (int ki = 0; ki < kernelSize; ki++) {
							for (int kj = 0; kj < kernelSize; kj++) {
								float pixel;
								int depthOffset = d * inputW * inputH;
								int rowOffset = stride * i + ki - padding;
								int colOffset = stride * j + kj - padding;
								if (rowOffset < 0 || colOffset < 0 || rowOffset >= inputW || colOffset >= inputH) {
									pixel = 0;
									idx[ki * kernelSize + kj] = 0;
								} else {
									pixel = A.get(depthOffset + rowOffset * inputH + colOffset, sample);
									idx[ki * kernelSize + kj] = depthOffset + rowOffset * inputH + colOffset;
								}
								list[ki * kernelSize + kj] = pixel;
							}
						}
						int max = max(list);
						img[d * (W * H) + i * W + j][sample] = list[max];
						// 标记max在原图中的位置
						index.put(sample, new ImmutablePair<>(d * (W * H) + i * W + j, idx[max]));
					}
				}
			}
		}
		this.A = new FloatMatrix(img);
		return this.A;
	}

	public int max(float[] l) {
		float m = l[0];
		int i = 0;
		for (int j=0; j<l.length; j++) {
			float f = l[j];
			if (f > m) {
				m = f;
				i = j;
			}
		}
		return i;
	}

	public FloatMatrix backward() {
		FloatMatrix delta;
		if (next == null) {
			delta = this.delta;
		} else {
			delta = next.delta;
		}
		int N = delta.columns;
		FloatMatrix img = FloatMatrix.zeros(inputD * inputW * inputH, N);
		for (Integer sample : index.keySet()) {
			for (Pair<Integer, Integer> tmp : index.get(sample)) {
				Integer deltaI = tmp.getLeft();
				Integer inputI = tmp.getRight();
				img.put(inputI, sample, delta.get(deltaI, sample));
			}
		}
		this.delta = img;
		return this.delta;
	}

	public void pullWeights() {
		index.clear();
		return;
	}
}

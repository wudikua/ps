package layer;

import com.google.common.collect.Lists;
import lombok.Data;
import org.jblas.FloatMatrix;
import store.KVStore;
import update.Updater;

import java.util.List;

@Data
public abstract class Layer {

	protected boolean isInput = false;

	protected FloatMatrix A;

	protected FloatMatrix delta;

	protected Layer next;

	protected List<Layer> inputs = Lists.newArrayList();

	protected Layer pre;

	protected int inputDims, outputDims;

	protected String name;

	protected KVStore kvStore = KVStore.ins();

	public Layer(){}

	public Layer(String name, int inputDims, int outputDims) {
		this.inputDims = inputDims;
		this.outputDims = outputDims;
		this.name = name;
	}

	// 正向
	public abstract FloatMatrix forward();

	// 反向
	public abstract FloatMatrix backward();

	// 同步参数
	public abstract void pullWeights();

	public Layer setInput(FloatMatrix A) {
		this.A = A;
		return this;
	}

	public Layer setNext(Layer l) {
		this.next = l;
		l.pre = this;
		return this;
	}

	public Layer setPre(Layer l) {
		this.pre = l;
		return this;
	}

	public Layer setInputs(List<Layer> l) {
		this.inputs = l;
		return this;
	}

	public Layer setIsInput(boolean input) {
		this.isInput = input;
		return this;
	}

	public boolean isInput() {
		return isInput;
	}
}

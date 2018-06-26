package layer;

import com.google.common.collect.Lists;
import lombok.Data;
import org.jblas.FloatMatrix;

import java.util.List;

@Data
public class ConcatLayer extends Layer {

	protected List<Layer> inputs = Lists.newArrayList();

	public ConcatLayer(String name, List<Layer> layers) {
		super();
		int outputDims = 0;
		for (Layer layer : layers) {
			outputDims += layer.getOutputDims();
		}
		this.inputs = layers;
		this.name = name;
		this.inputDims = outputDims;
		this.outputDims = outputDims;
	}

	public ConcatLayer(String name, int inputDims, int outputDims) {
		super(name, inputDims, outputDims);
	}

	public FloatMatrix forward() {
		this.A = inputs.get(0).A;
		for (int i=1; i<inputs.size(); i++) {
			Layer l = inputs.get(i);
			this.A = FloatMatrix.concatVertically(this.A, l.A);
		}
		return this.A;
	}

	public FloatMatrix backward() {
		this.delta = next.delta;
		int offset = 0;
		for (int i=0; i<inputs.size(); i++) {
			Layer l = inputs.get(i);
			l.backward();
			offset += l.A.rows;
		}
		return null;
	}

	@Override
	public void pullWeights() {

	}

}

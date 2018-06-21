package layer;

import activations.Relu;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.Data;
import org.jblas.FloatMatrix;
import store.KVStore;
import update.Updater;
import util.MatrixUtil;

import java.util.List;

@Data
public class EmbeddingLayer extends Layer {

	List<EmbeddingField> embeddingFields = Lists.newArrayList();

	public EmbeddingLayer(String name, int inputDims, int outputDims) {
		super(name, inputDims, outputDims);
	}

	public FloatMatrix forward() {
		FloatMatrix E = pre.A;
		float[][] EX = new float[embeddingFields.get(0).getOutputDims() * embeddingFields.size()][E.columns];
		// 对每个域做embedding
		for (int i=0; i< embeddingFields.size(); i++) {
			EmbeddingField layer = embeddingFields.get(i);

			FloatMatrix embed = layer.forward(E.getRow(i).toArray());
			// 每个域embedding以后的结果合并到结果中
			MatrixUtil.appendRows(i * embeddingFields.get(0).getOutputDims(), EX, embed);
		}
		A =  new FloatMatrix(EX);
		return A;
	}

	public EmbeddingLayer build(int embeddingFieldNum, int embeddingSize) {
		for (int j = 0; j < embeddingFieldNum; j++) {
			EmbeddingField em = new EmbeddingField("emF" + j, 1, embeddingSize);
			em.setActivation(new Relu());
			embeddingFields.add(em);
		}
		return this;
	}

	public FloatMatrix backward() {
		this.delta = next.delta;
		int offset = 0;
		for (int i = 0; i< embeddingFields.size(); i++) {
			EmbeddingField embeddingField = embeddingFields.get(i);
			embeddingField.backward(offset, delta);
			offset += embeddingField.getOutputDims();
		}
		// embedding 必须是最后一层
		return this.delta;
	}

	public void pullWeights() {
		for (EmbeddingField embeddingField : embeddingFields) {
			embeddingField.clear();
		}
	}

	public EmbeddingLayer setEmbeddingFields(List<EmbeddingField> fields) {
		this.embeddingFields = fields;
		return this;
	}

}

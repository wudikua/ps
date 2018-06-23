package net;

import com.google.common.collect.Maps;
import context.Context;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.Data;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import store.MyKey;

import javax.sound.midi.MetaMessage;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

@Data
public class PSClient {

	static Logger logger = LoggerFactory.getLogger(PSClient.class);

	ManagedChannel channel;

	net.PSGrpc.PSFutureStub stub;

	public PSClient() {
		this(Context.psHost, Context.psPort);
	}

	public PSClient(String host, int port) {
		channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext(true).build();
		stub = net.PSGrpc.newFutureStub(channel).withCompression("gzip");
	}

	public void close() {
		channel.shutdown();
	}

	// 从参数服务器获取参数
	public FloatMatrix get(String key) {
		Matrix matrix = Matrix.newBuilder().setKey(key).build();
		try {
			GetMessage result = stub.get(GetMessage.newBuilder()
					.setMeta(RequestMeta.newBuilder().setHost(Context.host).build())
					.setWeights(matrix).build()).get();
			if (result.getResp().getEc() != 200) {
				if (result.getResp().getEc() != 204) {
					logger.info("get error {} key:{}", result.getResp().getEm(), key);
				}
				return null;
			}
			float[] data = new float[result.getWeights().getDataCount()];
			for (int i=0; i<result.getWeights().getDataCount(); i++) {
				data[i] = result.getWeights().getData(i);
			}
			FloatMatrix tmp = new FloatMatrix();
			tmp.data = data;
			tmp.rows = result.getWeights().getRow();
			tmp.columns = result.getWeights().getCols();
			tmp.length = tmp.rows * tmp.columns;
			if (!key.contains("emF")) {
				logger.info("get key {} len {} rows {} cols {}", key, data.length, tmp.rows, tmp.columns);
			}
			return tmp;
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		return null;
	}

	public Map<String, FloatMatrix> getList(List<String> keys) {
		logger.info("getList keys {}", keys.toString().substring(0, 100));
		Map<String, FloatMatrix> result = Maps.newHashMap();
		GetListMessage.Builder request = GetListMessage.newBuilder();
		for (String key : keys) {
			request.addWeights(Matrix.newBuilder().setKey(key));
		}
		try {
			GetListMessage resp = stub.getList(request.build()).get();
			if (resp.getResp().getEc() != 200) {
				return null;
			}
			for (int i=0; i<resp.getWeightsCount(); i++) {
				if (resp.getWeights(i).getDataCount() == 0) {
					result.put(resp.getWeights(i).getKey(), null);
					continue;
				}
				float[] data = new float[resp.getWeights(i).getDataCount()];
				for (int j=0; j<resp.getWeights(i).getDataCount(); j++) {
					data[j] = resp.getWeights(i).getData(j);
				}
				FloatMatrix tmp = new FloatMatrix();
				tmp.data = data;
				tmp.rows = resp.getWeights(i).getRow();
				tmp.columns = resp.getWeights(i).getCols();
				tmp.length = tmp.rows * tmp.columns;
				result.put(resp.getWeights(i).getKey(), tmp);
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		return result;
	}

	// 更新matrix到参数服务器 replace为false 重复key不替换
	public FloatMatrix update(String key, FloatMatrix weights, boolean replace) {
		Matrix.Builder matrixBuilder = Matrix.newBuilder().setKey(key);
		matrixBuilder.setRow(weights.rows);
		matrixBuilder.setCols(weights.columns);
		for (int i=0; i<weights.data.length; i++) {
			matrixBuilder.addData(weights.data[i]);
		}
		try {
			UpdateMessage result = stub.upsert(UpdateMessage.newBuilder()
					.setMeta(RequestMeta.newBuilder().setHost(Context.host).build())
					.setReplace(replace)
					.setWeights(matrixBuilder).build()).get();
			if (result.getResp().getEc() != 200) {
				logger.info("get error {}", result.getResp().getEm());
				return null;
			}
			if (!result.getWeights().getUpdate()) {
				// 成功替换
				return weights;
			}
			// 没有替换，获取到新的weights
			float[] data = new float[result.getWeights().getDataCount()];
			for (int i=0; i<result.getWeights().getDataCount(); i++) {
				data[i] = result.getWeights().getData(i);
			}
			FloatMatrix tmp = new FloatMatrix();
			tmp.data = data;
			tmp.rows = result.getWeights().getRow();
			tmp.columns = result.getWeights().getCols();
			tmp.length = tmp.rows * tmp.columns;
			if (!key.contains("emF")) {
				logger.info("update key {} len {} rows {} cols {}", key, data.length, tmp.rows, tmp.columns);
			}
			return tmp;
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		return null;
	}

	public Map<String, FloatMatrix> updateList(Map<String, FloatMatrix> updates, boolean replace) {
		UpdateListMessage.Builder request = UpdateListMessage.newBuilder().setReplace(replace);
		for (String key : updates.keySet()) {
			FloatMatrix weights = updates.get(key);
			Matrix.Builder matrixBuilder = Matrix.newBuilder().setKey(key);
			matrixBuilder.setRow(weights.rows);
			matrixBuilder.setCols(weights.columns);
			for (int i=0; i<weights.data.length; i++) {
				matrixBuilder.addData(weights.data[i]);
			}
			request.addWeights(matrixBuilder);
		}
		try {
			UpdateListMessage resp = stub.upsertList(request.build()).get();
			for (int i=0; i<resp.getWeightsCount(); i++) {
				Matrix m = resp.getWeights(i);
				if (!m.getUpdate()) {
					continue;
				}
				// 没有替换，获取到新的weights
				float[] data = new float[m.getDataCount()];
				for (int j=0; j<m.getDataCount(); j++) {
					data[j] = m.getData(j);
				}
				FloatMatrix tmp = new FloatMatrix();
				tmp.data = data;
				tmp.rows = m.getRow();
				tmp.columns = m.getCols();
				tmp.length = tmp.rows * tmp.columns;
				updates.put(m.getKey(), tmp);
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		return updates;
	}

	// 推送梯度
	public void push(String key, FloatMatrix gradient, String updaterKey, boolean async) {
		Matrix.Builder matrixBuilder = Matrix.newBuilder().setKey(key);
		matrixBuilder.setRow(gradient.rows);
		matrixBuilder.setCols(gradient.columns);
		for (float f : gradient.data) {
			matrixBuilder.addData(f);
		}
		try {
			Future<GradientMessage> future = stub.push(GradientMessage.newBuilder()
					.setMeta(RequestMeta.newBuilder().setHost(Context.host).build())
					.setUpdaterKey(updaterKey).setGradient(matrixBuilder).build());
			if (!async) {
				GradientMessage result = future.get();
				if (result.getResp().getEc() != 200) {
					logger.info("get error {}", result.getResp().getEm());
					return;
				}
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		return;
	}

	// 阻塞等待ps 进入下一轮训练
	public void barrier() {
		try {
			stub.barrier(BarrierMessage.newBuilder()
					.setMeta(RequestMeta.newBuilder().setHost(Context.host).build()).build()).get();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
	}
}

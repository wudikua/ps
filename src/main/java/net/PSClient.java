package net;

import context.Context;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.Data;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sound.midi.MetaMessage;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

@Data
public class PSClient {

	static Logger logger = LoggerFactory.getLogger(PSClient.class);

	ManagedChannel channel;

	PSGrpc.PSFutureStub stub;

	public PSClient() {
		this(Context.psHost, Context.psPort);
	}

	public PSClient(String host, int port) {
		channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext(true).build();
		stub = PSGrpc.newFutureStub(channel).withCompression("gzip");
	}

	public void close() {
		channel.shutdown();
	}

	// 从参数服务器获取参数
	public FloatMatrix get(String key) {
		Matrix matrix = Matrix.newBuilder().setKey(key).build();
		try {
			GetMessage result = stub.get(GetMessage.newBuilder()
					.setMeta(RequestMeta.newBuilder().setHost("localhost").build())
					.setWeights(matrix).build()).get();
			if (result.getResp().getEc() != 200) {
				logger.info("get error {} key:{}", result.getResp().getEm(), key);
				return null;
			}
			float[] data = new float[result.getWeights().getDataList().size()];
			int i=0;
			for (Float f : result.getWeights().getDataList()) {
				data[i++] = f;
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

	// 更新matrix到参数服务器 replace为false 重复key不替换
	public FloatMatrix update(String key, FloatMatrix weights, boolean replace) {
		Matrix.Builder matrixBuilder = Matrix.newBuilder().setKey(key).setReplace(replace);
		matrixBuilder.setRow(weights.rows);
		matrixBuilder.setCols(weights.columns);
		for (int i=0; i<weights.data.length; i++) {
			matrixBuilder.addData(weights.data[i]);
		}
		try {
			UpdateMessage result = stub.upsert(UpdateMessage.newBuilder()
					.setMeta(RequestMeta.newBuilder().setHost(Context.host).build())
					.setWeights(matrixBuilder).build()).get();
			if (result.getResp().getEc() != 200) {
				logger.info("get error {}", result.getResp().getEm());
				return null;
			}
			if (result.getReplace()) {
				// 成功替换
				return weights;
			}
			// 没有替换，获取到新的weights
			float[] data = new float[result.getWeights().getDataList().size()];
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

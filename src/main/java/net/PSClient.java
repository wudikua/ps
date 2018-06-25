package net;

import com.google.common.collect.Maps;
import context.Context;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.Data;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.MatrixUtil;

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
			FloatMatrix tmp = MatrixUtil.ProtoMatrix_2_FloatMatrix(result.getWeights());
			if (!key.contains("emF")) {
				logger.info("get key {} len {} rows {} cols {}", key, tmp.data.length, tmp.rows, tmp.columns);
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
		logger.debug("getList keys {}, len {}", keys.subList(0, Math.min(keys.size(), 10)), keys.size());
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
				result.put(resp.getWeights(i).getKey(), MatrixUtil.ProtoMatrix_2_FloatMatrix(resp.getWeights(i)));
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
		try {
			UpdateMessage result = stub.upsert(UpdateMessage.newBuilder()
					.setMeta(RequestMeta.newBuilder().setHost(Context.host).build())
					.setReplace(replace)
					.setWeights(MatrixUtil.FloatMatrix_2_ProtoMatrix(key, weights)).build()).get();
			if (result.getResp().getEc() != 200) {
				logger.info("get error {}", result.getResp().getEm());
				return null;
			}
			if (!result.getWeights().getUpdate()) {
				// 成功替换
				return weights;
			}
			// 没有替换，获取到新的weights
			FloatMatrix tmp = MatrixUtil.ProtoMatrix_2_FloatMatrix(result.getWeights());
			if (!key.contains("emF")) {
				logger.info("update key {} len {} rows {} cols {}", key, tmp.data.length, tmp.rows, tmp.columns);
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
			request.addWeights(MatrixUtil.FloatMatrix_2_ProtoMatrix(key, weights));
		}
		try {
			UpdateListMessage resp = stub.upsertList(request.build()).get();
			for (int i=0; i<resp.getWeightsCount(); i++) {
				Matrix m = resp.getWeights(i);
				if (!m.getUpdate()) {
					continue;
				}
				// 没有替换，获取到新的weights
				FloatMatrix tmp = MatrixUtil.ProtoMatrix_2_FloatMatrix(m);
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
		try {
			Future<GradientMessage> future = stub.push(GradientMessage.newBuilder()
					.setMeta(RequestMeta.newBuilder().setHost(Context.host).build())
					.setUpdaterKey(updaterKey)
					.setGradient(MatrixUtil.FloatMatrix_2_ProtoMatrix(key, gradient)).build());
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

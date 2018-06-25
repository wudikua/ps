package net;

import com.google.common.collect.Maps;
import context.Context;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import lombok.Data;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import store.KVStore;
import update.Updater;
import util.MatrixUtil;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

@Data
public class PServer implements net.PSGrpc.PS, Runnable {

	static Logger logger = LoggerFactory.getLogger(PServer.class);

	static final Resp success = Resp.newBuilder().setEc(200).setEm("").build();

	private Server server;

	private Map<String, Updater> updaterMap = Maps.newConcurrentMap();

	private KVStore store;

	private Map<String, FloatMatrix> storeFloatMatrix = Maps.newConcurrentMap();

	private int workerNum;

	private final AtomicLong globalStep = new AtomicLong(0);

	private final AtomicLong workerStep = new AtomicLong(0);

	private Map<String, String> updateKeys = Maps.newConcurrentMap();

	private Executor updateThread = Executors.newSingleThreadExecutor();

	private Resp error(int ec, String em) {
		return Resp.newBuilder().setEc(ec).setEm(em).build();
	}

	public PServer(int port, int workerNum) {
		server = ServerBuilder.forPort(port).addService(net.PSGrpc.bindService(this)).build();
		store = KVStore.ins();
		this.workerNum = workerNum;
		updateThread.execute(this);
	}

	public void start() {
		try {
			server.start();
			server.awaitTermination();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} finally {
			close();
		}
	}

	public void close() {
		server.shutdown();
	}

	public void get(GetMessage request, StreamObserver<GetMessage> responseObserver) {
		if (!request.getWeights().getKey().contains("emF")) {
			logger.info("request {}", request.getWeights().getKey());
		}
		FloatMatrix result = store.get(request.getWeights().getKey());
		if (result == null) {
			GetMessage resp = GetMessage.newBuilder().setResp(error(204, "null weights")).build();
			responseObserver.onNext(resp);
			responseObserver.onCompleted();
			return;
		}
        Matrix.Builder m = MatrixUtil.FloatMatrix_2_ProtoMatrix(request.getWeights().getKey(), result);
		if (!request.getWeights().getKey().contains("emF")) {
			logger.info("key {} data len {}", request.getWeights().getKey(), result.data.length);
		}
		if (request.getWeights().getKey().contains("emF13.28305")) {
			logger.info("data key{} {}", request.getWeights().getKey(), result.data);
		}
		GetMessage resp = GetMessage.newBuilder().setWeights(m.build()).setResp(success).build();
		responseObserver.onNext(resp);
		responseObserver.onCompleted();
	}

	@Override
	public void getList(GetListMessage request, StreamObserver<GetListMessage> responseObserver) {
		GetListMessage.Builder resp = GetListMessage.newBuilder();
		for (int i=0; i<request.getWeightsCount(); i++) {
			String key = request.getWeights(i).getKey();
			FloatMatrix result = store.get(key);
			if (result == null) {
				// add empty weights when null
				resp.addWeights(MatrixUtil.FloatMatrix_2_ProtoMatrix(key, null));
				continue;
			}
			resp.addWeights(MatrixUtil.FloatMatrix_2_ProtoMatrix(key, result));
		}
		resp.setResp(success).build();
		responseObserver.onNext(resp.build());
		responseObserver.onCompleted();
	}

	public void upsert(UpdateMessage request, StreamObserver<UpdateMessage> responseObserver) {
		if (!request.getWeights().getKey().contains("emF")) {
			logger.info("insert {}", request.getWeights().getKey());
		}
		boolean update = true;
		UpdateMessage.Builder resp = UpdateMessage.newBuilder().setResp(success);
		String key = request.getWeights().getKey();
		FloatMatrix exists = store.get(key);
		if (exists == null || request.getReplace()) {
			update = false;
			exists = MatrixUtil.ProtoMatrix_2_FloatMatrix(request.getWeights());
			if (!request.getWeights().getKey().contains("emF")) {
				logger.info("put {} len {} rows {} cols {}",
						key, exists.data.length, exists.rows, exists.columns);
			}
			store.put(key, exists);
		}
		Matrix.Builder m = MatrixUtil.FloatMatrix_2_ProtoMatrix(key, exists);
		m.setUpdate(update);
		resp.setWeights(m.build());
		responseObserver.onNext(resp.build());
		responseObserver.onCompleted();
	}

	@Override
	public void upsertList(UpdateListMessage request, StreamObserver<UpdateListMessage> responseObserver) {
		UpdateListMessage.Builder resp = UpdateListMessage.newBuilder();
		for (int i=0; i<request.getWeightsCount(); i++) {
			boolean update = true;
			String key = request.getWeights(i).getKey();
			FloatMatrix exists = store.get(key);
			if (exists == null || request.getReplace()) {
				update = false;
				exists = MatrixUtil.ProtoMatrix_2_FloatMatrix(request.getWeights(i));
				store.put(request.getWeights(i).getKey(), exists);
			}
			Matrix.Builder m = MatrixUtil.FloatMatrix_2_ProtoMatrix(key, exists);
			m.setUpdate(update);
			resp.addWeights(m);
		}
		resp.setResp(success).build();
		responseObserver.onNext(resp.build());
		responseObserver.onCompleted();
	}

	public void push(GradientMessage request, StreamObserver<GradientMessage> responseObserver) {
		String key = request.getGradient().getKey();
		if (!request.getGradient().getKey().contains("emF")) {
			logger.info("update {}", key);
		}
		Updater updater = updaterMap.get(request.getUpdaterKey());
		if (updater == null) {
			logger.error("updater {} is null", request.getUpdaterKey());
			responseObserver.onNext(GradientMessage.newBuilder().setResp(error(500, "updater is null")).build());
			responseObserver.onCompleted();
			return;
		}

		store.sum(key, MatrixUtil.ProtoMatrix_2_FloatMatrix(request.getGradient()));
		if (!updateKeys.containsKey(key)) {
			updateKeys.put(key, request.getUpdaterKey());
		}
		GradientMessage.Builder resp = GradientMessage.newBuilder();
		responseObserver.onNext(resp.build());
		responseObserver.onCompleted();
	}

	public void run() {
		while (true) {
			synchronized (this) {
				try {
					// 等待barrier的通知
					logger.info("wait workers finish");
					this.wait();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				logger.info("start update all params for term {}", globalStep.get());
				// 更新key
				for (String key :updateKeys.keySet()) {
					Updater updater = updaterMap.get(updateKeys.get(key));
					if (!key.contains("emF")) {
						logger.info("update key {}", key);
					}
					store.update(updater, key);
				}
				globalStep.incrementAndGet();
				synchronized (globalStep) {
					// 通知所有barrier
					globalStep.notifyAll();
				}
			}
		}
	}

	public void barrier(BarrierMessage request, StreamObserver<BarrierMessage> responseObserver) {
		long step = globalStep.get();
		long wStep = workerStep.incrementAndGet();
		logger.info("barrier step {} wStep {}", step, wStep);
		// 等待其他worker
		logger.info("wait for other worker");
		while (wStep < step + this.workerNum) {
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			wStep = workerStep.get();
		}
		logger.info("barrier step {} wStep {}", step, wStep);
		// 通知更新线程
		synchronized (this) {
			// 通知更新线程
			logger.info("notify ps worker");
			this.notify();
		}
		// 等待更新线程执行完毕的通知
		logger.info("wait for ps worker update start");
		while (wStep + this.workerNum < step) {
			synchronized (globalStep) {
				try {
					globalStep.wait(100);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				step = globalStep.get();
			}
		}
		logger.info("wait for ps worker update end");
		BarrierMessage.Builder resp = BarrierMessage.newBuilder();
		resp.setResp(success);
		responseObserver.onNext(resp.build());
		responseObserver.onCompleted();
	}

}

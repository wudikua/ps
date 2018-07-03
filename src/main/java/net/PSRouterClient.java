package net;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.collect.Multimap;
import context.Context;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.Data;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.MatrixUtil;

import java.util.List;
import java.util.Map;
import java.util.concurrent.*;

@Data
public class PSRouterClient extends PSClient {

	static Logger logger = LoggerFactory.getLogger(PSRouterClient.class);

	List<PSClient> clients = Lists.newArrayList();

	Router router;

	ExecutorService executor = Executors.newCachedThreadPool();

	public PSRouterClient(Router router) {
		this.router = router;
		String[] addrs = Context.psAddrs.split(",");
		for (String addr : addrs) {
			String[] hostport = addr.split(":");
			String host = hostport[0];
			int port = Integer.parseInt(hostport[1]);
			clients.add(new PSClient(host, port));
		}
	}

	public PSRouterClient() {
		this(new Mod(Context.psAddrs.split(",").length));
	}

	public void close() {
		for (PSClient c : clients) {
			c.close();
		}
	}

	// 从参数服务器获取参数
	public FloatMatrix get(String key) {
		PSClient c = clients.get(router.shard(key));
		return c.get(key);
	}

	public Map<String, FloatMatrix> getList(List<String> keys) {
		Multimap<PSClient, String> request = HashMultimap.create();
		Map<String, FloatMatrix> response = Maps.newHashMap();
		for (String key : keys) {
			request.put(clients.get(router.shard(key)), key);
		}
		List<Future<Map<String, FloatMatrix>>> futures = Lists.newArrayList();
		for (PSClient c : request.keySet()) {
			futures.add(executor.submit(new Callable<Map<String, FloatMatrix>>() {
				@Override
				public Map<String, FloatMatrix> call() throws Exception {
					return c.getList(Lists.newArrayList(request.get(c)));
				}
			}));
		}
		for (Future<Map<String, FloatMatrix>> f : futures) {
			try {
				response.putAll(f.get());
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		return response;
	}

	// 更新matrix到参数服务器 replace为false 重复key不替换
	public FloatMatrix update(String key, FloatMatrix weights, boolean replace) {
		PSClient c = clients.get(router.shard(key));
		return c.update(key, weights, replace);
	}

	public Map<String, FloatMatrix> updateList(Map<String, FloatMatrix> updates, boolean replace) {
		Multimap<PSClient, Pair<String, FloatMatrix>> request = HashMultimap.create();
		Map<String, FloatMatrix> response = Maps.newHashMap();
		for (String key : updates.keySet()) {
			request.put(clients.get(router.shard(key)), new MutablePair<>(key, updates.get(key)));
		}
		List<Future<Map<String, FloatMatrix>>> futures = Lists.newArrayList();
		for (PSClient c : request.keySet()) {
			futures.add(executor.submit(new Callable<Map<String, FloatMatrix>>() {
				@Override
				public Map<String, FloatMatrix> call() throws Exception {
					Map<String, FloatMatrix> params = Maps.newHashMap();
					for (Pair<String, FloatMatrix> p : request.get(c)) {
						params.put(p.getKey(), p.getValue());
					}
					return c.updateList(params, replace);
				}
			}));
		}
		for (Future<Map<String, FloatMatrix>> f : futures) {
			try {
				response.putAll(f.get());
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
		return response;
	}

	// 推送梯度
	public void push(String key, FloatMatrix gradient, String updaterKey, boolean async) {
		PSClient c = clients.get(router.shard(key));
		c.push(key, gradient, updaterKey, async);
	}

	// 阻塞等待ps 进入下一轮训练
	public void barrier() {
		List<Future> futures = Lists.newArrayList();
		for (PSClient c : clients) {
			futures.add(executor.submit(new Callable() {
				@Override
				public Void call() throws Exception {
					c.barrier();
					return null;
				}
			}));
		}
		for (Future f : futures) {
			try {
				f.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} catch (ExecutionException e) {
				e.printStackTrace();
			}
		}
	}
}

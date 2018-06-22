package store;


import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.SettableFuture;
import context.Context;
import lombok.Data;
import net.PSClient;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import update.Updater;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Condition;
import java.util.function.Supplier;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;

public class KVStore {

	private static Logger logger = LoggerFactory.getLogger(KVStore.class);

	private static KVStore ins = new KVStore();

	// 单机版使用
	private Map<String, FloatMatrix> store = Maps.newConcurrentMap();

	// PS架构使用
	private ThreadLocal<PSClient> client;

	// 异步初始化一批权重
	private Set<MyKey> asyncGet = Sets.newConcurrentHashSet();

	private Map<String, FloatMatrix> sum = Maps.newConcurrentMap();
	private Map<String, AtomicLong> sumCnt = Maps.newConcurrentMap();

	private KVStore() {
		if (!Context.isPServer() && Context.isDistributed()) {
			client = new ThreadLocal<>().withInitial(new Supplier<PSClient>() {
				@Override
				public PSClient get() {
					return new PSClient();
				}
			});
		}
	}

	public static KVStore ins() {
		return ins;
	}

	public FloatMatrix getList(List<MyKey> keys) {
		// @TODO 调用底层批量获取
		return null;
	}

	public FloatMatrix get(String key) {
		if (!Context.isPServer() && Context.isDistributed()) {
			return client.get().get(key);
		}
		return store.get(key);
	}

	public synchronized FloatMatrix get(String key, Callable<FloatMatrix> init) {
		if (Context.isDistributed()) {
			// 分布式
			if (store.containsKey(key)) {
				return store.get(key);
			}
			FloatMatrix m = client.get().get(key);
			if (m != null) {
				store.put(key, m);
				return m;
			}
			m = create(key, init);
			store.put(key, m);
			return m;
		}
		// 单机
		if (store.containsKey(key)) {
			return store.get(key);
		}
		return create(key, init);
	}

	public void put(String key, FloatMatrix val) {
		if (Context.isDistributed() && !Context.isPServer()) {
			throw new RuntimeException("KVStore put method only used in ps server");
		}
		store.put(key, val);
	}

	private FloatMatrix create(String key, Callable<FloatMatrix> init) {
		if (!Context.isPServer() && Context.isDistributed()) {
			try {
				FloatMatrix m = init.call();
				return client.get().update(key, m, false);
			} catch (Exception e) {
				e.printStackTrace();
			}
			return null;
		}
		if (store.containsKey(key)) {
			return store.get(key);
		}
		try {
			FloatMatrix m = init.call();
			store.put(key, m);
			return m;
		} catch (Exception e) {
			e.printStackTrace();
		}
		return null;
	}

	public synchronized void sum(String key, FloatMatrix val) {
		if (!sum.containsKey(key)) {
			sum.put(key, val);
			sumCnt.put(key, new AtomicLong(1));
		} else {
			sum.get(key).addi(val);
			sumCnt.get(key).incrementAndGet();
		}
	}

	public void update(Updater updater, String key) {
		FloatMatrix g = sum.get(key).divi(sumCnt.get(key).get());
		if (key.contains("emF13.28305")) {
			logger.info("key {} data {}  g {}", key, store.get(key).data, g);
		}
		if (Context.isPServer() || Context.isStandalone()) {
			updater.update(key, store.get(key), g);
		} else {
			client.get().push(key, g, updater.getName(), true);
		}
		// 分布式版本BSP更新需要barrier
		if (!Context.isPServer() && Context.isDistributed()) {
			logger.info("worker barrier waiting begin");
			client.get().barrier();
			logger.info("worker barrier waiting end");
		}
	}

	public void update(Updater updater) {
		for (String key : sum.keySet()) {
			FloatMatrix g = sum.get(key).divi(sumCnt.get(key).get());
			if (Context.isPServer() || Context.isStandalone()) {
				// 单机版在本地更新
				updater.update(key, store.get(key), g);
			} else {
				// 推送梯度
				client.get().push(key, g, updater.getName(), true);
			}
		}
		// 分布式版本BSP更新需要barrier
		if (!Context.isPServer() && Context.isDistributed()) {
			logger.info("worker barrier waiting begin");
			client.get().barrier();
			logger.info("worker barrier waiting end");
		}
	}

	public void clear() {
		sum.clear();
		sumCnt.clear();
		if (Context.isDistributed() && !Context.isPServer()) {
			logger.info("clear worker cache weights");
			store.clear();
		}
	}
}

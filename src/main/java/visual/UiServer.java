package visual;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import context.Context;
import data.TestDataSet;
import fi.iki.elonen.NanoHTTPD;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import lombok.Data;
import net.PServer;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

@Data
public class UiServer extends NanoHTTPD implements UiServerGrpc.UiServer {

	static Logger logger = LoggerFactory.getLogger(UiServer.class);

	ObjectMapper objectMapper = new ObjectMapper();

	ConcurrentHashMap<String, List<Float>> xs = new ConcurrentHashMap<>();

	ConcurrentHashMap<String, List<Float>> ys = new ConcurrentHashMap<>();

    Server server;

	public UiServer() throws IOException {
		super(8888);
        Context.init();
        start(NanoHTTPD.SOCKET_READ_TIMEOUT, false);
		System.out.println("\nRunning! Point your browsers to http://localhost:8888/ \n");
        server = ServerBuilder.forPort(Context.uiPort).addService(UiServerGrpc.bindService(this)).build();
        try {
            server.start();
            logger.info("start ui grpc port {}", Context.uiPort);
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

	public static void main(String args[]) {
		try {
			new UiServer();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public Response serve(IHTTPSession session) {
		Map<String, String> params = session.getParms();
		if ("data".equals(params.get("act"))) {
			// /act=data&setp=1&key=fc0,fc1
			String keys = params.get("key");
			int step = Integer.parseInt(params.get("step"));
			Collection<String> set = xs.keySet();
			if (StringUtils.isNotBlank(keys)) {
				set = Lists.newArrayList(keys.split(","));
			}
			Map<String, Object> result = Maps.newHashMap();
			for (String key : set) {
				List<Float> x = xs.get(key);
				List<Float> y = ys.get(key);
				if (x == null || y == null || x.size() < step) {
					continue;
				}
				Map<String, Object> tmp = Maps.newHashMap();
				synchronized (x) {
                    tmp.put("x", x.subList(step, x.size()));
                }
                synchronized (y) {
                    tmp.put("y", y.subList(step, y.size()));
                }
				result.put(key, tmp);
			}
			try {
				return newFixedLengthResponse(objectMapper.writeValueAsString(result));
			} catch (JsonProcessingException e) {
				return newFixedLengthResponse(e.getMessage());
			}
		} else if("list_graph".equals(params.get("act"))) {
			// /act=list_graph
			Map<String, Object> result = Maps.newHashMap();
			result.put("graphs", ys.keySet());
			try {
				return newFixedLengthResponse(objectMapper.writeValueAsString(result));
			} catch (JsonProcessingException e) {
				return newFixedLengthResponse(e.getMessage());
			}
		} else {
			return newFixedLengthResponse(TestDataSet.readToString(UiServer.class.getResource("").getPath()+"../../../src/main/resources/web/index.html"));
		}
	}

	@Override
	public void plot(PlotMessage request, StreamObserver<PlotMessage> responseObserver) {
	    logger.debug("plot {}", request.getId());
		String key = request.getId();
		List<Float> y = ys.putIfAbsent(key, Lists.newArrayList());
		List<Float> x = xs.putIfAbsent(key, Lists.newArrayList());
        if (x == null) {
            x = xs.putIfAbsent(key, Lists.newArrayList());
        }
        if (y == null) {
            y = ys.putIfAbsent(key, Lists.newArrayList());
        }
		assert y != null;
		synchronized (y) {
			y.addAll(request.getData().getYList());
		}
		assert x != null;
		synchronized (x) {
			x.addAll(request.getData().getXList());
		}
		responseObserver.onNext(PlotMessage.newBuilder().build());
		responseObserver.onCompleted();
	}
}

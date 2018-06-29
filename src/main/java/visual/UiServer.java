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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Map;
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
//		new Thread(new Runnable() {
//			@Override
//			public void run() {
//				AtomicInteger l = new AtomicInteger(0);
//				while (true) {
//					synchronized (this) {
//						List<Float> x = xs.putIfAbsent("loss", Lists.newArrayList());
//						if (x == null) {
//							x = xs.putIfAbsent("loss", Lists.newArrayList());
//						}
//						List<Float> y = ys.putIfAbsent("loss", Lists.newArrayList());
//						if (y == null) {
//							y = ys.putIfAbsent("loss", Lists.newArrayList());
//						}
//						x.add((float) l.getAndIncrement());
//						y.add(ThreadLocalRandom.current().nextFloat());
//
//                        List<Float> x2 = xs.putIfAbsent("auc", Lists.newArrayList());
//                        if (x2 == null) {
//                            x2 = xs.putIfAbsent("auc", Lists.newArrayList());
//                        }
//                        List<Float> y2 = ys.putIfAbsent("auc", Lists.newArrayList());
//                        if (y2 == null) {
//                            y2 = ys.putIfAbsent("auc", Lists.newArrayList());
//                        }
//                        x2.add((float) l.getAndIncrement());
//                        y2.add(ThreadLocalRandom.current().nextFloat());
//					}
//					try {
//						Thread.sleep(1000);
//					} catch (InterruptedException e) {
//						e.printStackTrace();
//					}
//				}
//			}
//		}).start();

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
			int step = Integer.parseInt(params.get("step"));
			Map<String, Object> result = Maps.newHashMap();
			for (String key : xs.keySet()) {
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
		} else {
			return newFixedLengthResponse(TestDataSet.readToString(UiServer.class.getResource("").getPath()+"../../../src/main/resources/web/index.html"));
		}
	}

	@Override
	public void plot(PlotMessage request, StreamObserver<PlotMessage> responseObserver) {
	    logger.info("plot {}", request.getId());
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

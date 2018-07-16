package visual;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import context.Context;
import fi.iki.elonen.NanoHTTPD;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import lombok.Data;
import net.PServer;
import org.apache.commons.lang.math.NumberUtils;
import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

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
		try {
			Map<String, String> params = session.getParms();
			Map<String, String> body = new HashMap<String, String>();
			session.parseBody(body);
			if ("data".equals(params.get("act"))) {
				// /act=data&key=fc0,fc1
				String keys = params.get("key");
				Map<String, Float> step = Maps.newHashMap();
				try {
					step = objectMapper.readValue(body.get("postData"), Map.class);
				} catch (IOException e) {
					e.printStackTrace();
				}
				Collection<String> set = xs.keySet();
				if (StringUtils.isNotBlank(keys)) {
					set = Lists.newArrayList(keys.split(","));
				}
				Map<String, Object> result = Maps.newHashMap();
				for (String key : set) {
					List<Float> x = xs.get(key);
					List<Float> y = ys.get(key);
					if (x == null || y == null) {
						continue;
					}
					Map<String, List<Float>> tmp = Maps.newHashMap();
					tmp.put("x", Lists.newArrayList());
					tmp.put("y", Lists.newArrayList());
					for (int i = 0; i < y.size(); i++) {
						if (!step.containsKey(key) || y.get(i) > NumberUtils.toFloat(String.valueOf(step.get(key)), 0f)) {
							tmp.get("x").add(x.get(i));
							tmp.get("y").add(y.get(i));
						}
					}
					result.put(key, tmp);
				}
				return newFixedLengthResponse(objectMapper.writeValueAsString(result));
			} else if ("list_graph".equals(params.get("act"))) {
				// /act=list_graph
				Map<String, Object> result = Maps.newHashMap();
				result.put("graphs", ys.keySet());
				return newFixedLengthResponse(objectMapper.writeValueAsString(result));
			} else {
				return newFixedLengthResponse(readToString(UiServer.class.getResource("").getPath() + "../../../src/main/resources/web/index.html"));
			}
		} catch (Exception e) {
			logger.error("http error", e);
			return newFixedLengthResponse(e.getMessage());
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

	public static String readToString(String fileName) {
		String encoding = "UTF-8";
		File file = new File(fileName);
		Long filelength = file.length();
		byte[] filecontent = new byte[filelength.intValue()];
		try {
			FileInputStream in = new FileInputStream(file);
			in.read(filecontent);
			in.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		try {
			return new String(filecontent, encoding);
		} catch (UnsupportedEncodingException e) {
			System.err.println("The OS does not support " + encoding);
			e.printStackTrace();
			return null;
		}
	}
}

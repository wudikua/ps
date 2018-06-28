package visual;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import data.TestDataSet;
import fi.iki.elonen.NanoHTTPD;
import lombok.Data;

import java.io.IOException;
import java.util.Map;

@Data
public class UiServer extends NanoHTTPD {

	ObjectMapper objectMapper = new ObjectMapper();

	public UiServer() throws IOException {
		super(8888);
		start(NanoHTTPD.SOCKET_READ_TIMEOUT, false);
		System.out.println("\nRunning! Point your browsers to http://localhost:8888/ \n");
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
		if ("loss".equals(params.get("act"))) {
			Map<String, Object> maps = Maps.newConcurrentMap();
			maps.put("x", Lists.newArrayList(1, 2, 3));
			maps.put("y", Lists.newArrayList(Math.random(), Math.random(), Math.random()));
			try {
				return newFixedLengthResponse(objectMapper.writeValueAsString(maps));
			} catch (JsonProcessingException e) {
				return newFixedLengthResponse(e.getMessage());
			}
		} else {
			return newFixedLengthResponse(TestDataSet.readToString(UiServer.class.getResource("").getPath()+"../../../src/main/resources/web/index.html"));
		}
	}

}

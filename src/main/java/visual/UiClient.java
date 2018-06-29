package visual;

import context.Context;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import net.PSClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ExecutionException;

public class UiClient {
    static Logger logger = LoggerFactory.getLogger(UiClient.class);

    ManagedChannel channel;

    UiServerGrpc.UiServerFutureStub stub;

    private static UiClient client = new UiClient();

    public static UiClient ins() {
        return client;
    }

    public UiClient() {
        this(Context.uiHost, Context.uiPort);
    }

    public UiClient(String host, int port) {
        channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext(true).build();
        stub = UiServerGrpc.newFutureStub(channel).withCompression("gzip");
    }

    public void close() {
        channel.shutdown();
    }

    public void plot(String id, float x, float y) {
        stub.plot(PlotMessage.newBuilder().setId(id).setData(Plot.newBuilder().addX(x).addY(y)).build());
    }
}

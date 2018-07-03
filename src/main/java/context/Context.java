package context;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

public class Context {
	public static ThreadLocal<Boolean> training = new ThreadLocal<Boolean>();

	public static ThreadLocal<AtomicLong> term = new ThreadLocal<>().withInitial(new Supplier<AtomicLong>() {
		@Override
		public AtomicLong get() {
			return new AtomicLong(0);
		}
	});

	public static long nTermDump;

	public static volatile boolean finish;

	public static volatile boolean dump;

	public static enum Mode {
		STANDALONE, DISTRIBUTED
	}

	public static volatile AtomicLong step = new AtomicLong(0);

	public static volatile boolean isPs;

	public static volatile boolean isPsAsync;

	public static volatile int workerNum;

	public static int thread;

	public static volatile int psPort;

	public static volatile String psHost;

	public static volatile String psAddrs;

	public static volatile int uiPort;

	public static volatile String uiHost;

	public static volatile Mode mode;

	private static volatile boolean inited = false;

	public static String host;

	public static void init() {
		if (inited) {
			return;
		}
		inited = true;
		if ("dist".equals(System.getProperty("mode", "stand"))) {
			mode = Mode.DISTRIBUTED;
		} else {
			mode = Mode.STANDALONE;
		}
		nTermDump = Integer.parseInt((System.getProperty("nTermDump", "20")));
		finish = false;
		dump = false;
		thread = Integer.parseInt(System.getProperty("thread", String.valueOf(Runtime.getRuntime().availableProcessors())));
		isPs = "1".equals(System.getProperty("ps", "0"));
		isPsAsync = "1".equals(System.getProperty("isPsAsync", "0"));
		workerNum = Integer.parseInt((System.getProperty("workerNum", "1")));
		psPort = Integer.parseInt((System.getProperty("psPort", "8890")));
		psHost = System.getProperty("psHost", "localhost");
		psAddrs = System.getProperty("psAddrs", "localhost:8890");
		uiPort = Integer.parseInt((System.getProperty("uiPort", "8990")));
		uiHost = System.getProperty("uiHost", "localhost");
		try {
			host = InetAddress.getLocalHost().getHostName();
			if (host.indexOf(".momo.com") > 0) {
				host = host.substring(0, host.indexOf(".momo.com"));
			}
		} catch (UnknownHostException e) {
			e.printStackTrace();
		}
	}

	public static boolean isPServer() {
		return isPs;
	}

	public static boolean isDistributed() {
		return mode.equals(Mode.DISTRIBUTED);
	}

	public static boolean isStandalone() {
		return mode.equals(Mode.STANDALONE);
	}
}

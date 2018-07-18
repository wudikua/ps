package data;

import com.google.common.collect.Lists;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.*;

public abstract class DataSet implements Runnable {

	static Logger logger = LoggerFactory.getLogger(DataSet.class);

    protected Parser parser;

    protected BlockingQueue<Map<String, FloatMatrix>> queue;

    protected ExecutorService executor;

    protected DataSource source;

    protected int batch;

    protected int thread;


	protected volatile boolean eof = false;

    public DataSet(Parser parser, DataSource source, int batch, int thread) {
        this.parser = parser;
        this.source = source;
        this.batch = batch;
        this.thread = thread;
        // 1倍 预先填充
        this.queue = new ArrayBlockingQueue<>(thread * 2);
		start();
    }

    public Map<String, FloatMatrix> next() {
		if (queue.isEmpty() && eof) {
			return null;
		}
		try {
			return queue.poll(3, TimeUnit.SECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		return null;
	}

	public boolean hasNext() {
		if (queue.isEmpty() && eof) {
			return false;
		}
		return true;
	}

	public void reset() {
		executor.shutdownNow();
		queue.clear();
		source.reset();
		eof = false;
		start();
	}

    public void start() {
		executor = Executors.newFixedThreadPool(thread);
        for (int i=0; i<thread; i++) {
            executor.execute(this);
        }
    }

    public void run() {
        while (!eof) {
            try {
                List<List<Feature>> dataList = Lists.newArrayList();
                for (int i=0; i<batch; i++) {
                    String line = source.readLine();
					if (line == null) {
						eof = true;
						logger.info("read eof");
						break;
					}
                    List<Feature> features = parser.parse(line);
                    dataList.add(features);
                }
				if (dataList.isEmpty()) {
					continue;
				}
				logger.debug("add batch data to queue");
                queue.put(parseFeature(dataList));
            } catch (Exception e) {
                // ignore
            }
        }
    }

    public abstract Map<String, FloatMatrix> parseFeature(List<List<Feature>> dataList);
}

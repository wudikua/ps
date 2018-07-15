package data;

import com.google.common.collect.Lists;
import org.jblas.FloatMatrix;

import java.io.BufferedReader;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public abstract class DataSet implements Runnable {

    protected Parser parser;

    protected Queue<Map<String, FloatMatrix>> queue;

    protected ExecutorService executor;

    protected DataSource source;

    protected int batch;

    protected int thread;

    public DataSet(Parser parser, DataSource source, int batch, int thread) {
        this.parser = parser;
        this.source = source;
        this.batch = batch;
        this.thread = thread;
        // 1倍 预先填充
        this.queue = new ArrayBlockingQueue<>(thread * batch * 2);
        this.executor = Executors.newFixedThreadPool(thread);
    }

    public Map<String, FloatMatrix> next() {
        return queue.poll();
    }

    public void start() {
        for (int i=0; i<thread; i++) {
            executor.execute(this);
        }
    }

    public void run() {
        while (true) {
            try {
                List<List<Feature>> dataList = Lists.newArrayList();
                for (int i=0; i<batch; i++) {
                    String line = source.readLine();
                    List<Feature> features = parser.parse(line);
                    dataList.add(features);
                }
                queue.add(parseFeature(dataList));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public abstract Map<String, FloatMatrix> parseFeature(List<List<Feature>> dataList);
}

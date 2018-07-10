package evaluate;

import context.Context;
import loss.Loss;
import model.Model;
import org.jblas.FloatMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import train.PredictThread;
import visual.UiClient;

import java.math.BigDecimal;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class LossSurface {

    static Logger logger = LoggerFactory.getLogger(LossSurface.class);

    // 一组训练集
    Map<String, FloatMatrix> data;

    // 一组真实值
    FloatMatrix y;

    // 损失函数
    Loss lossFunc;

    // 模型
    Model model;

	ExecutorService executor = Executors.newSingleThreadExecutor();

    public LossSurface(Map<String, FloatMatrix> data, FloatMatrix y, Loss lossFunc, Model model) {
        this.data = data;
        this.y = y;
        this.lossFunc = lossFunc;
        this.model = model;
    }

    // w = scale * w_init + (1-scale) * w_final
    float scale = 0.1f;

    float min = -2;

    float max = 2;

    public void plot() {
        Context.status = Context.Stat.LOSS_SURFACE_EVAL;
        for (float i=min; i<max; i+=scale) {
            Context.weightsScale = i;
			try {
				FloatMatrix p = executor.submit(new PredictThread(0, model, data)).get();
				float val = lossFunc.forward(p, y);
				logger.info("plot {} {}", new BigDecimal(i).setScale(2, 4).floatValue(), val);
				UiClient.ins().plot("loss_surface_" + Context.step, val, new BigDecimal(i).setScale(2, 4).floatValue());
			} catch (Exception e) {
				e.printStackTrace();
			}
        }
    }
}

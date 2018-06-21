package evaluate;

import lombok.Data;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.util.*;

@Data
public class AUC {

    private double posNum; //正
    private double negNum;

    private Pair<Double, Double> dataList[];

    public AUC(List<Pair<Double, Double>> dataList) {
        this.dataList = new MutablePair[dataList.size()];
        dataList.toArray(this.dataList);
    }

	public AUC(float[] p, float[] y) {
		List<Pair<Double, Double>> data = new ArrayList<Pair<Double, Double>>();
		for (int i=0; i<y.length; i++) {
			data.add(new MutablePair<Double, Double>((double) p[i], (double) y[i]));
		}
        this.dataList = new MutablePair[data.size()];
        data.toArray(this.dataList);
	}

    // 排序样本统计
    private void sampleCount() {
        Arrays.sort(this.dataList, new Comparator<Pair<Double, Double>>() {
            @Override
            public int compare(Pair<Double, Double> o1, Pair<Double, Double> o2) {
                return o1.getKey().compareTo(o2.getKey());
            }
        });

        for (Pair<Double, Double> pair : this.dataList) {
            double y = pair.getValue();
            if (y > 0.0) {
                this.posNum += 1;
            } else {
                this.negNum += 1;
            }
        }
    }

    //计算坐标点
    private List<Pair<Double, Double>> getCoordinatePoint() {
        List<Pair<Double, Double>> result = new ArrayList<Pair<Double, Double>>();
        double tp = 0, fp = 0;
        for (int i = this.dataList.length - 1; i >= 0; i--) {
            Pair<Double, Double> pair = this.dataList[i];
            double y = pair.getValue();
            if (y > 0.0) {
                fp += 1;
            } else {
                tp += 1;
            }
            result.add(new MutablePair<Double, Double>(tp / this.posNum, fp / this.negNum));
        }
        return result;
    }

    public double calculate() {
        sampleCount(); //排序 & 样本统计
        List<Pair<Double, Double>> coordinate = getCoordinatePoint();

        double prev = 0;
        double auc = 0;
        for (Pair<Double, Double> pair : coordinate) {
            double x = pair.getKey();
            double y = pair.getValue();
            if(x != prev) {
                auc += (x - prev) * y;
                prev = x;
            }
        }
        return auc;
    }
}

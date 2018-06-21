package loss;

import org.jblas.FloatMatrix;

public class MSE implements Loss {

    @Override
    public float forward(FloatMatrix predict, FloatMatrix label) {
        float sum = 0;
        for (int i=0; i<predict.columns; i++) {
            float p =  predict.get(0, i);
            float l =  label.get(0, i);
            sum += 0.5 * Math.pow((l - p),2);
        }
        return sum / predict.columns;
    }

    @Override
    public FloatMatrix backward(FloatMatrix predict, FloatMatrix label) {
        FloatMatrix delta = predict.dup();
        for (int i=0; i<predict.columns; i++) {
            float p =  predict.get(0, i);
            float l =  label.get(0, i);
            delta.data[predict.index(0, i)] = (p - l);
        }
        return delta;
    }
}

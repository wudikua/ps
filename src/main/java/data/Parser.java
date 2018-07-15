package data;

import org.jblas.FloatMatrix;

import java.util.List;
import java.util.Map;

public interface Parser {
    /**
     * 返回每一列 第0列是label
     */
    public List<Feature> parse(String line);
}

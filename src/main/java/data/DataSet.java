package data;


import org.apache.commons.lang3.tuple.Pair;
import org.jblas.FloatMatrix;

import java.util.Iterator;

public interface DataSet<E> extends Iterator<E> {
	public Pair<FloatMatrix, FloatMatrix> nextBatch();
}

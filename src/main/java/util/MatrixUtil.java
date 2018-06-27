package util;

import net.Matrix;
import org.apache.commons.lang3.RandomUtils;
import org.jblas.FloatMatrix;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by mengjun on 18/5/23.
 */
public class MatrixUtil {

	public static String pretty(FloatMatrix f) {
		StringBuffer sb = new StringBuffer();
		sb.append("\n");
		for (int i=0; i<f.rows; i++) {
			sb.append("[");
			for (int j=0; j<f.columns; j++) {
				sb.append(f.get(i, j)).append(",");
			}
			sb.append("]\n");
		}
		return sb.toString();
	}

	public static FloatMatrix hash(FloatMatrix m, int size) {
		FloatMatrix result = m.dup();
		for (int i=0; i<result.data.length; i++) {
			result.data[i] = result.data[i] % size;
		}
		return result;
	}

	public static FloatMatrix randE(int row, int col) {
		float[][] result = new float[row][col];
		int[] r = new int[]{0,1};
		for (int i=0; i<row; i++) {
			for (int j=0; j<col; j++) {
				result[i][j] = r[j%2];
			}
		}
		return new FloatMatrix(result);
	}

	public static FloatMatrix randY(int row, int col) {
		float[][] result = new float[row][col];
		int[] r = new int[]{0,1};
		for (int i=0; i<row; i++) {
			for (int j=0; j<col; j++) {
				result[i][j] = r[j%2];
			}
		}
		return new FloatMatrix(result);
	}

	public static FloatMatrix rand(int row, int col) {
		float[][] result = new float[row][col];
		for (int i=0; i<row; i++) {
			for (int j=0; j<col; j++) {
				if (ThreadLocalRandom.current().nextInt(1) == 0) {
					result[i][j] = (float)ThreadLocalRandom.current().nextGaussian();
				} else {
					result[i][j] = (float)(0-ThreadLocalRandom.current().nextGaussian());
				}
			}
		}
		return new FloatMatrix(result);
	}

	public static FloatMatrix rand(int row, int col, float max) {
		float[][] result = new float[row][col];
		for (int i=0; i<row; i++) {
			for (int j=0; j<col; j++) {
				if (RandomUtils.nextInt(0, 2) == 0) {
					result[i][j] = RandomUtils.nextFloat(0, max);
				} else {
					result[i][j] = 0-RandomUtils.nextFloat(0, max);
				}
			}
		}
		return new FloatMatrix(result);
	}

	public static void appendRows(int offset, float[][] base, FloatMatrix append) {
		for (int i=0; i<append.rows; i++) {
			for (int j = 0; j < append.columns; j++) {
				base[offset + i][j] = append.get(i, j);
			}
		}
	}

	public static Matrix.Builder FloatMatrix_2_ProtoMatrix(String key, FloatMatrix matrix) {
		Matrix.Builder m = Matrix.newBuilder().setKey(key);
		if (matrix == null) {
			return m;
		}
		float[] data = matrix.data;
		for (int i=0; i<data.length; i++) {
			m.addData(data[i]);
		}
		m.setRow(matrix.rows);
		m.setCols(matrix.columns);
		return m;
	}

	public static FloatMatrix ProtoMatrix_2_FloatMatrix(Matrix m) {
		float[] data = new float[m.getDataCount()];
		for (int i=0; i<m.getDataCount(); i++) {
			data[i] = m.getData(i);
		}
		FloatMatrix tmp = new FloatMatrix();
		tmp.data = data;
		tmp.rows = m.getRow();
		tmp.columns = m.getCols();
		tmp.length = tmp.rows * tmp.columns;
		return tmp;
	}

}

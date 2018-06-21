package util;

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

}

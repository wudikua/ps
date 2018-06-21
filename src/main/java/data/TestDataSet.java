package data;


import lombok.Data;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.jblas.FloatMatrix;

import java.io.*;
import java.util.Iterator;

public class TestDataSet {

	@Data
	public static class MatrixData {
		FloatMatrix X;
		FloatMatrix E;
		FloatMatrix Y;

		public MatrixData(FloatMatrix x, FloatMatrix e, FloatMatrix y) {
			X = x;
			E = e;
			Y = y;
		}
	}
	
	public static String readToString(String fileName) {  
        String encoding = "UTF-8";  
        File file = new File(fileName);  
        Long filelength = file.length();  
        byte[] filecontent = new byte[filelength.intValue()];  
        try {  
            FileInputStream in = new FileInputStream(file);  
            in.read(filecontent);  
            in.close();  
        } catch (FileNotFoundException e) {  
            e.printStackTrace();  
        } catch (IOException e) {  
            e.printStackTrace();  
        }  
        try {  
            return new String(filecontent, encoding);  
        } catch (UnsupportedEncodingException e) {  
            System.err.println("The OS does not support " + encoding);  
            e.printStackTrace();  
            return null;  
        }  
    }

	public static class FileIterator implements Iterator<MatrixData> {

		int pos = 0;
		int N;

		String filename;
		String lines[];

		public FileIterator(String filename, int N) {
			this.filename = filename;
			this.N = N;
			lines = readToString(TestDataSet.class.getClassLoader().getResource(filename).getPath()).split("\n");
		}

		public boolean hasNext() {
			return pos < lines.length;
		}

		public void reset() {
			pos = 0;
		}

		public MatrixData next() {
			float[][] E = new float[23][N];
			float[][] X = new float[45][N];
			float[][] P = new float[1][N];
			int col = 0;
			for (int n =0; pos<lines.length && n < N; pos++) {
				n++;
				String line = lines[pos];
				try {
					String[] cols = line.split(" ");
					// 从前向后取23个离散特征
					for (int i = 1; i < 24; i++) {
						E[i-1][col] = Float.parseFloat(cols[i].split(":")[0]);
					}
					// 从后向前取45个连续特征
					for (int i = 24; i < 69; i++) {
						X[i-24][col] = Float.parseFloat(cols[i].split(":")[1]);
					}
					// 初始化label
					P[0][col] = Integer.parseInt(cols[0]);
					col++;
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			return new MatrixData(new FloatMatrix(X), new FloatMatrix(E), new FloatMatrix(P));
		}

		@Override
		public void remove() {

		}

	}

	public static FileIterator fromIteratorFile(String filename) throws IOException {
		if (System.getProperty("batch") == null) {
			System.setProperty("batch", System.getenv("batch"));
		}
		return new FileIterator(filename, Integer.parseInt(System.getProperty("batch", "3000")));
	}

	public static synchronized Pair<MatrixData, Boolean> fromStream(BufferedReader reader, int N) throws IOException {
		int nn = 0;
		float[][] E = new float[23][N];
		float[][] X = new float[45][N];
		float[][] P = new float[1][N];
		int col = 0;
		boolean remind = true;
		while (++nn <= N) {
			String line = reader.readLine();
			if (line == null) {
				remind = false;
				break;
			}
			try {
				String[] cols = line.split(" ");
				// 从前向后取23个离散特征
				for (int i = 1; i < 24; i++) {
					E[i-1][col] = Float.parseFloat(cols[i].split(":")[0]);
				}
				// 从后向前取45个连续特征
				for (int i = 24; i < 69; i++) {
					X[i-24][col] = Float.parseFloat(cols[i].split(":")[1]);
				}
				// 初始化label
				P[0][col] = Integer.parseInt(cols[0]);
				col++;
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println(line);
			}
		}
		return new ImmutablePair(new MatrixData(new FloatMatrix(X), new FloatMatrix(E), new FloatMatrix(P)), remind);
	}

	public static MatrixData fromFile(String filename) throws IOException {
		int nn = 0;
		int N = 100;
		String lines[] = readToString(TestDataSet.class.getClassLoader().getResource(filename).getPath()).split("\n");
		float[][] E = new float[23][N];
		float[][] X = new float[45][N];
		float[][] P = new float[1][N];
		int col = 0;
		for (String line : lines) {
			if (++nn > N) {
				break;
			}
			try {
				String[] cols = line.split(" ");
				// 从前向后取23个离散特征
				for (int i = 1; i < 24; i++) {
					E[i-1][col] = Float.parseFloat(cols[i].split(":")[0]);
				}
				// 从后向前取45个连续特征
				for (int i = 24; i < 69; i++) {
					X[i-24][col] = Float.parseFloat(cols[i].split(":")[1]);
				}
				// 初始化label
				P[0][col] = Integer.parseInt(cols[0]);
				col++;
			} catch (Exception e) {
				e.printStackTrace();
				System.out.println(line);
			}
		}
		return new MatrixData(new FloatMatrix(X), new FloatMatrix(E), new FloatMatrix(P));
	}
}

package data;

import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public abstract class DataSource {
    // 数据偏移
    protected int offset = 0;
    protected int idx = 0;
    // 读步长
    protected int step = 1;

	protected ReadWriteLock resetLock = new ReentrantReadWriteLock();

    public void reset() {
		try {
			resetLock.writeLock().lock();
			idx = 0;
			resetInternal();
		} finally {
			resetLock.writeLock().unlock();
		}
	}

    public String readLine() {
		try {
			resetLock.readLock().lock();
			// seek to offset
			while (idx <= offset) {
				String line = readLineInternal();
				idx++;
				if (idx - 1 == offset) {
					return line;
				}
			}
			// run step;
			String line = null;
			for (int i = 0; i < step; i++) {
				line = readLineInternal();
				idx++;
			}
			return line;
		} finally {
			resetLock.readLock().unlock();
		}
	}

    public abstract String readLineInternal();

	public abstract void resetInternal();
}


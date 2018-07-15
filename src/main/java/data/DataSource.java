package data;

public abstract class DataSource {
    // 数据偏移
    protected int offset;
    protected int idx = 0;
    // 读步长
    protected int step = 1;

    public abstract void reset();

    public String readLine() {
        // seek to offset
        while(idx<=offset) {
            String line = readLineInternal();
            idx++;
            if (idx - 1 == offset) {
                return line;
            }
        }
        // run step;
        String line = null;
        for (int i=0; i<step; i++) {
            line = readLineInternal();
            idx++;
        }
        return line;
    }

    public abstract String readLineInternal();
}


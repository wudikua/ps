package data;

import java.io.*;

public class FileSource extends DataSource {

    File file;

    BufferedReader reader;

    public FileSource(File file) {
        this.file = file;
        try {
            this.reader = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void reset() {
        if (this.reader != null) {
            try {
                this.reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            this.reader = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    @Override
    public String readLineInternal() {
        if (reader == null) {
            throw new RuntimeException("file source reader is null, check file path");
        }
        try {
            return reader.readLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}

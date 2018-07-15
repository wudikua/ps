package data;

import com.google.common.collect.Lists;
import lombok.Data;

import java.util.List;

@Data
public class Feature {

    // 特征的indx
    long idx;

    // 特征值
    Object o;

    public Feature(long idx, Object o) {
        this.idx = idx;
        this.o = o;
    }

    public int toI() {
        return (int)o;
    }

    public String toS() {
        return (String)o;
    }

    public float toF() {
        return (float)o;
    }
}

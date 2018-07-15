package data;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.commons.lang.StringUtils;

import java.util.List;
import java.util.Map;

public class LibsvmParser implements Parser {
    // key 是第几组
    @Override
    public List<Feature> parse(String line) {
        if (StringUtils.isBlank(line)) {
            return Lists.newArrayList();
        }
        List<Feature> result = Lists.newArrayList();
        String[] cols = line.split(" ");
        result.add(new Feature(0, Float.parseFloat(cols[0])));
        for (int i=1; i<cols.length; i++) {
            String[] pair = cols[i].split(":");
            result.add(new Feature(Long.parseLong(pair[0]), Float.parseFloat(pair[1])));
        }
        return result;
    }
}

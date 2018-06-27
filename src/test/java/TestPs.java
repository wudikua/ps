import context.Context;
import net.PServer;
import org.apache.commons.lang.StringUtils;
import org.junit.Test;
import update.AdamUpdater;
import update.FtrlUpdater;
import update.Updater;

public class TestPs {

	@Test
	public void start() {
		Context.init();
		Context.isPs = true;
		Updater updater = new AdamUpdater(0.005, 0.9, 0.999, Math.pow(10, -8));
		Updater ftrl = new FtrlUpdater(0.005f, 1f, 0.001f, 0.001f);
		PServer server = new PServer(8890, 1);
		server.getUpdaterMap().put(updater.getName(), updater);
		server.getUpdaterMap().put(ftrl.getName(), ftrl);
		server.start();
	}

	@Test
	public void test() {
		String s = "adam@alfa:1@beta1:2@beta2:3@";
		System.out.println(StringUtils.substringBetween(s, "beta2:", "@"));
	}
}

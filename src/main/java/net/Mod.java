package net;


public class Mod implements Router {

	int n;

	public Mod(int n) {
		this.n = n;
	}

	@Override
	public int shard(String key) {
		return key.hashCode() % n;
	}
}

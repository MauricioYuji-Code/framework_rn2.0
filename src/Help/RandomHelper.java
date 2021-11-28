package Help;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class RandomHelper {

    //Randomizar entre um n min e max
    public static double randomizeNumberMinMax(int min, int max) {
        ThreadLocalRandom tlr = ThreadLocalRandom.current();
        return Math.floor(tlr.nextDouble(min, max) * 100) / 100;
    }

    //Randomizar entre 0 e 1
    public static double randomizeNumberZeroOne() {
        Random random = new Random();
        return Math.floor(random.nextDouble() * 100) / 100;
    }

}

package mnist;

import Help.Input;

import java.util.ArrayList;

public class MNISTInput implements Input {

    private int min = 0;
    private int max = 255;
    private int[] data;

    public MNISTInput() {

    }

    @Override
    public double[] getInput(ArrayList<Number> in, float max, float min) {
        double[] result = new double[in.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = in.get(i).doubleValue() / (double) (max - min);
            System.out.println(result[i]);
        }

        return result;
    }

}

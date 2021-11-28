package mnist;

import Help.Input;

import java.util.ArrayList;

public class MNISTInput{

    private int min = 0;
    private int max = 255;
    private int[] data;

    public MNISTInput() {

    }

    public double[] getInput(ArrayList<Number> in, float max, float min) {
        double[] result = new double[in.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = in.get(i).doubleValue() / (double) (max - min);
            System.out.println(result[i]);
        }

        return result;
    }

    public double[] getInputArray(MnistData in, float max, float min) {
        double[] result = new double[in.getSizeOfArray()];
        for (int i = 0; i < result.length; i++) {
            result[i] = in.getValueOfArray(i) / (double) (max - min);
//            System.out.println(result[i]);
        }

        return result;
    }

    public double[] getInputArray(MnistData in) {
        double[] result = new double[in.getSizeOfArray()];
        for (int i = 0; i < result.length; i++) {
            result[i] = in.getValueOfArray(i);
            System.out.println(result[i]);
        }

        return result;
    }

}

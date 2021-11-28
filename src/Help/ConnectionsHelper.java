package Help;

import core.Layer;

import java.util.ArrayList;

public class ConnectionsHelper {

    public static ArrayList<Double> fillBiasAccordingToNeurons(Layer layer) {
        ArrayList<Double> bias = new ArrayList<>();
        for (int i = 0; i < layer.getNeuronsCount(); i++) {
            bias.add(RandomHelper.randomizeNumberZeroOne());
        }
        return bias;
    }

    public static Layer fillWeightsAccordingToNeurons(Layer initiation, Layer destination) {
        int i = 0;
        while (i < initiation.getNeuronsCount()) {
            for (int j = 0; j < destination.getNeuronsCount(); j++) {
                initiation.getNeurons().get(i).addInputConnection(destination.getNeurons().get(j), RandomHelper.randomizeNumberMinMax(-1, 1));
            }
            i++;
        }
        return initiation;
    }

    public static Layer fillWeightsAccordingToNeuronsIncludingWeight(Layer initiation, Layer destination, double weigthValue) {
        int i = 0;
        while (i < initiation.getNeuronsCount()) {
            for (int j = 0; j < destination.getNeuronsCount(); j++) {
                initiation.getNeurons().get(i).addInputConnection(destination.getNeurons().get(j), weigthValue);
            }
            i++;
        }
        return initiation;
    }

}

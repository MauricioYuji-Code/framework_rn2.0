package Help;

import core.Layer;
import core.NeuralNetwork;

import java.util.ArrayList;

public class CalculatorHelper {

    public static void triggerSummation(Layer initiation, Layer destination, ArrayList<Double> bias, NeuralNetwork neuralNetwork, String label) {
        int indicatorLayer = 0;
        double resultSumNeuron = 0;
        while (indicatorLayer < destination.getNeuronsCount()) {
            for (int i = 0; i < initiation.getNeuronsCount(); i++) {
                resultSumNeuron += initiation.getNeurons().get(i).getNetInput() * initiation.getNeurons().get(i).getInputConnections().get(indicatorLayer).getWeight().getValue();
            }
            resultSumNeuron += bias.get(indicatorLayer);
            neuralNetwork.startFunctionActivation(indicatorLayer, resultSumNeuron, destination, label);
            indicatorLayer++;
            resultSumNeuron = 0;
        }
    }
}

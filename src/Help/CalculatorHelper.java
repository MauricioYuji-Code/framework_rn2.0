package Help;

import core.Layer;
import core.NeuralNetwork;
import network.FunctionActivation;

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

    public static ArrayList<Double> fillListWithSigmoidDerivativesOutput(Layer layer, ArrayList<Double> list, NeuralNetwork neuralNetwork) {
        for (int i = 0; i < layer.getNeuronsCount(); i++) {
            list.add(FunctionActivation.sigmoidDer(layer.getNeurons().get(i).getOutput()));
        }
        return list;
    }

    public static ArrayList<Double> multplyListByListAccordingToNumberOfNeurons(Layer layer, ArrayList<Double> listA, ArrayList<Double> listB) {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < layer.getNeuronsCount(); i++) {
            result.add(listA.get(i) * listB.get(i));
        }
        return result;
    }

    public static ArrayList<Double> scaleListAccordingToNumberOfNeurons(Layer layer, ArrayList<Double> list, double number) {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < layer.getNeuronsCount(); i++) {
            result.add(list.get(i) * number);
        }
        return result;
    }

    public static ArrayList<Double> addListByListAccordingToNumberOfNeurons(Layer layer, ArrayList<Double> listA, ArrayList<Double> listB) {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < layer.getNeuronsCount(); i++) {
            result.add(listA.get(i) + listB.get(i));
        }
        return result;
    }

    public static double round(double n) {
        if (n > 0.98) {
            return 1;
        } else if (n < 0.04) {
            return 0;
        }
        return n;
    }
}

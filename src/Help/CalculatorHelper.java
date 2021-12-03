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

    public static ArrayList<Double> calculateHiddenLayerError(Layer layer, ArrayList<Double> list) {
        ArrayList<Double> result = new ArrayList<>();
        int connections = 0;
        double valueHiddenError = 0;
        while (connections < layer.getNeuronsCount()) {
            for (int i = 0; i < layer.getNeurons().get(connections).getInputConnections().size(); i++) {
                valueHiddenError += layer.getNeurons().get(connections).getInputConnections().get(i).getWeight().getValue() * list.get(i);
            }
            result.add(connections, valueHiddenError);
            connections++;
        }
        return result;
    }

    public static ArrayList<Double> calculateDeltaWeight(Layer layer, ArrayList<Double> list, NeuralNetwork neuralNetwork) {
        ArrayList<Double> result = new ArrayList<>();
        int connections = 0;
        while (connections < layer.getNeuronsCount()) {
            for (int i = 0; i < layer.getNeurons().get(connections).getInputConnections().size(); i++) {
                result.add(neuralNetwork.deltaWeigthCalc2(list.get(i), layer.getNeurons().get(connections).getNetInput()));
            }
            connections++;
        }
        return result;
    }

    public static Layer calculateNewWeight(Layer layer, ArrayList<Double> list, NeuralNetwork neuralNetwork) {
        int connections = 0;
        while (connections < layer.getNeuronsCount()) {
            for (int i = 0; i < layer.getNeurons().get(connections).getInputConnections().size(); i++) {
                layer.getNeurons().get(connections).getInputConnections().get(i).getWeight().setValue(neuralNetwork.newWeightCalc(layer.getNeurons().get(connections).getInputConnections().get(i).getWeight().getValue(), list.get(connections)));
            }
            connections++;
        }
//        Layer layerResult = layer;
        return layer;
    }

    //Todo Deixar genérico
    public static ArrayList<Double> calculateDerivativesOfTheOutput(Layer layer, ArrayList<Double> list) {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < layer.getNeuronsCount(); i++) {
            result.add(i, FunctionActivation.sigmoidDer(list.get(i)));
        }
        return result;
    }

    public static ArrayList<Double> calculateDerivativesOfTheOutputNoReferenceList(Layer layer) {
        ArrayList<Double> result = new ArrayList<>();
        for (int i = 0; i < layer.getNeuronsCount(); i++) {
            result.add(i, FunctionActivation.sigmoidDer(layer.getNeurons().get(i).getOutput()));
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

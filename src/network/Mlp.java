package network;

import Help.CalculatorHelper;
import Help.ConnectionsHelper;
import Help.Input;
import core.Layer;
import core.NeuralNetwork;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;


public class Mlp extends NeuralNetwork implements Serializable, Input {
    private double learningRate;
    private double predict;
    private double bias;
    private Layer input;
    private Layer hidden;
    private Layer output;
    private ArrayList<Layer> hiddens;
    private ArrayList<Double> deltaOW;
    private ArrayList<Double> deltaHW;
    private FunctionActivationData functionActivation;
    private ArrayList<Double> outputs = new ArrayList<>();
    private ArrayList<double[]> samples;
    //teste
    private double target = 1;
    private double noTarget = 0;
    private ArrayList<Double> bias_h = new ArrayList<>();
    private ArrayList<Double> bias_o = new ArrayList<>();
    //Variáveis auxiliares
    private int samplesCount = 1;
    private int epoch = 1;
    private int trainingCount = 0;
    private int samplePositionList = 0; //samplePositionOf
    private int predictPositionList = 0;
    private int round = 0;
    private int nTraining = 0;
    private int numberOfTrainings;
    private int trainings = 0;
    //Teste
    private ArrayList<Double> predicts = new ArrayList<>();
    private boolean end = true;
    //provisório
    ArrayList<Double> errorListO = new ArrayList<>();
    private String predictType = "classifier"; //Classificador ou Regressão


    public Mlp(double learningRate, double predict, double bias) {
        this.learningRate = learningRate;
        this.predict = predict;
        this.bias = bias;
    }

    public Mlp(double learningRate, ArrayList<Double> predicts) {
        this.learningRate = learningRate;
        this.predicts = predicts;
    }

    public Mlp(double learningRate, double predict) {
        this.learningRate = learningRate;
        this.predict = predict;
    }

    @Override
    public void setStructure(Type type, int nLayer, int nNeuron) {
        String aux = type.getTypeName();
        if (nLayer <= 0 || nNeuron <= 0) {
            System.out.println("Ops, algo errado no numero de camadas/neurônios");
            System.exit(0);
        }
        switch (aux) {
            case "INPUT":
                System.out.println("Camada sendo estruturada " + type);
                Layer input = new Layer(nNeuron);
                this.input = input;
                break;
            case "OUTPUT":
                System.out.println("Camada sendo estruturada " + type);
                if (nNeuron == 1) {
                    predictType = "degression";
                }
                Layer output = new Layer(nNeuron);
                this.output = output;
                break;
            case "HIDDEN":
                System.out.println("Camada sendo estruturada " + type);
                if (nLayer > 1) {
                    hiddens = new ArrayList<>();
                    for (int i = 0; i <= nLayer; i++) {
                        Layer hidden = new Layer(nNeuron);
                        hiddens.add(i, hidden);
                    }
                }
                Layer hidden = new Layer(nNeuron);
                this.hidden = hidden;
                break;
            default:
                System.out.println("Você não utilizou nem input ou output como tipo da camada");
                System.exit(0);
        }
    }

    public void randomizeBias() {
        this.bias_h = ConnectionsHelper.fillBiasAccordingToNeurons(hidden);
        this.bias_o = ConnectionsHelper.fillBiasAccordingToNeurons(output);
    }

    @Override
    public void setInputValues(ArrayList inputValues) {
        this.samples = inputValues;
        double[] convertedData = convertData(samples, samplePositionList);
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).setInput(convertedData[i]);//Generalizar entradas
//            input.getNeurons().get(i).setInput(samples.get(samplePositionList)[i]);
//            System.out.println(input.getNeurons().get(i).getNetInput());
//            if (i % 28 == 0)
//                System.out.println();
        }
    }

    public double[] convertData(ArrayList<double[]> samples, int index) {
        return getInput(samples.get(index), Arrays.stream(samples.get(index)).max().getAsDouble(), Arrays.stream(samples.get(index)).min().getAsDouble());
    }

    public double getPredictValue(ArrayList predicts) {
        return (double) predicts.get(predictPositionList);
    }

    public void setData(double[] data) {
        System.out.println("passou por aqui");
        data = getInput(data, Arrays.stream(data).max().getAsDouble(), Arrays.stream(data).min().getAsDouble());
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).setInput(data[i]);
//            System.out.println(input.getNeurons().get(i).getNetInput());
        }
    }

    @Override
    public void connectNeuronIncludingWeigth(double weigthValue) {
        input = ConnectionsHelper.fillWeightsAccordingToNeuronsIncludingWeight(input, hidden, weigthValue);
        hidden = ConnectionsHelper.fillWeightsAccordingToNeuronsIncludingWeight(hidden, output, weigthValue);
        randomizeBias();
    }

    @Override
    public void connectNeuronIncludingRandomWeigth() {
        input = ConnectionsHelper.fillWeightsAccordingToNeurons(input, hidden);
        hidden = ConnectionsHelper.fillWeightsAccordingToNeurons(hidden, output);
        randomizeBias();
    }

    @Override
    public void training() {
        sum();
        switch (predictType) {
            case "classifier":
                while (!checkOutputsClassifier()) {
                    backpropagationTest();
                    activateCounters();
                    sum();
                    if (numberOfTrainings != 0 && numberOfTrainings == trainings) {
                        System.out.println("Fim da execução");
                        break;
                    }
                }
                break;
            case "degression":
                while (!checkOutputsDegression()) {
//                while (end) {
                    backpropagationTest();
                    activateCounters();
                    sum();
                    if (numberOfTrainings != 0 && numberOfTrainings == trainings) {
                        this.end = false;
                        System.out.println("Fim da execução");
                        break;
                    }
                }
                break;
        }
        if (end == true)
        checkNextSamples();
    }

    public void checkNextSamples() {
        if (samplesCount != samples.size()) {
            trainWiththeNextSamples();
        } else {
            generateNewEpoch();
        }
    }

    @Override
    public void start() {
        sum();
        if (predictType.equals("degression")) {
            System.out.println("Output: " + output.getNeurons().get(0).getOutput());
        }

        if (predictType.equals("classifier")) {
            checkOutputsClassifier();
        }
    }

    public void sum() {
        CalculatorHelper.triggerSummation(input, hidden, bias_h, this, "hidden");
        CalculatorHelper.triggerSummation(hidden, output, bias_o, this, "output");
    }

    public void startFunctionActivation(int ref, double aux, Layer layer, String label) {
        if (getFunctionActivation().name().equals("DEGRAU")) {
            if (label.equals("output")) {
                layer.getNeurons().get(ref).setOutput(FunctionActivation.degrau(aux));
            } else {
                layer.getNeurons().get(ref).setInput(FunctionActivation.degrau(aux));
            }
        } else if (getFunctionActivation().name().equals("SIGMOID")) {
            if (label.equals("output")) {
                layer.getNeurons().get(ref).setOutput(FunctionActivation.sigmoid(aux));
            } else
                layer.getNeurons().get(ref).setInput(FunctionActivation.sigmoid(aux));
        }
    }

    public boolean checkOutputsClassifier() {
        this.predictPositionList = samplePositionList;
        for (int i = 0; i < output.getNeuronsCount(); i++) {
            outputs.add(output.getNeurons().get(i).getOutput());
//            System.out.println("Posição da lista: " + i + " Saída armazenada: " + outputs.get(i));
        }
//        System.out.println("max da lista: " + Collections.max(outputs) + " position: " + outputs.indexOf(Collections.max(outputs)));
        if (outputs.indexOf(Collections.max(outputs)) == getPredictValue(predicts)) {
//            System.out.println("Retornou: " + outputs.indexOf(Collections.max(outputs)) + " e o valor esperado é: " + predict + " (SUCESSO)");
            System.out.println("Retornou: " + outputs.indexOf(Collections.max(outputs)) + " e o valor esperado é: " + getPredictValue(predicts) + " (SUCESSO)");
            outputs.clear();
            return true;
        } else {
//            System.out.println("Retornou: " + outputs.indexOf(Collections.max(outputs)) + " e o valor esperado é: " + predict + " (FALHA)");
            System.out.println("Retornou: " + outputs.indexOf(Collections.max(outputs)) + " e o valor esperado é: " + getPredictValue(predicts) + " (FALHA)");
            outputs.clear();
            return false;
        }
    }

    public boolean checkOutputsDegression() {
        if (predicts.isEmpty()) {
            if (Math.round(output.getNeurons().get(0).getOutput()) == predict) {
                System.out.println("retornou: " + Math.round(output.getNeurons().get(0).getOutput()) + " e o valor esperado é: " + predict + " (SUCESSO)");
                return true;
            } else {
                System.out.println("retornou: " + output.getNeurons().get(0).getOutput() + " e o valor esperado é: " + predict + " (FALHA)");
                return false;
            }
        } else {
            this.predictPositionList = samplePositionList;
            if (Math.round(output.getNeurons().get(0).getOutput()) == getPredictValue(predicts)) {
//            if (CalculatorHelper.round(output.getNeurons().get(0).getOutput()) == getPredictValue(predicts)) {
//                System.out.println("retornou: " + Math.round(output.getNeurons().get(0).getOutput()) + " e o valor esperado é: " + getPredictValue(predicts) + " (SUCESSO)");
//                System.out.println("retornou: " + Math.round(((output.getNeurons().get(0).getOutput()) * 100) / 100) + " e o valor esperado é: " + getPredictValue(predicts) + " (SUCESSO)");
//                System.out.println("retornou: " + output.getNeurons().get(0).getOutput() + " e o valor esperado é: " + getPredictValue(predicts) + " (SUCESSO)");
                System.out.println("retornou: " + CalculatorHelper.round(output.getNeurons().get(0).getOutput()) + " e o valor esperado é: " + getPredictValue(predicts) + " (SUCESSO)");
                return true;
            } else {
//                System.out.println("retornou: " + Math.round(output.getNeurons().get(0).getOutput()) + " e o valor esperado é: " + getPredictValue(predicts) + " (FALHA)");
//                System.out.println("retornou: " + Math.round(((output.getNeurons().get(0).getOutput()) * 100) / 100) + " e o valor esperado é: " + getPredictValue(predicts) + " (FALHA)");
//                System.out.println("retornou: " + output.getNeurons().get(0).getOutput() + " e o valor esperado é: " + getPredictValue(predicts) + " (FALHA)");
                System.out.println("retornou: " + CalculatorHelper.round(output.getNeurons().get(0).getOutput()) + " e o valor esperado é: " + getPredictValue(predicts) + " (FALHA)");
                return false;
            }
        }
    }

    public void trainWiththeNextSamples() {
        samplePositionList++;
        setInputValues(samples);
        samplesCount++;
        training();
    }

    public void generateNewEpoch() {
        this.epoch++;
        if (trainingCount > 0) {
            this.samplePositionList = 0;
            this.samplesCount = 0;
            this.round = 0;
            this.predictPositionList = 0;
            setInputValues(samples);
        }
        while (trainingCount > 0) {
            samplesCount++;
            this.trainingCount = 0;
            training();
        }
        this.samplePositionList = 0;
    }

    public void activateCounters() {
        trainingCount++;
        nTraining++;
        round++;
        trainings++;
    }

    public void backpropagationTest() {
        backpropagationOutputToHidden();
        backpropagationHiddenToAny();
    }

    //provisório
    public void backpropagationOutputToHidden() {
        //Oculta para a saída
        /*
        Formação do arraylist do calculo de cada erro (E)
         */
        //Generalizar
        errorListO = new ArrayList<>();
        if (predictType.equals("classifier")) {
            for (int i = 0; i < output.getNeuronsCount(); i++) {
                if (i == getPredictValue(predicts)) {
                    errorListO.add(target - output.getNeurons().get(i).getOutput());
                }
                errorListO.add(noTarget - output.getNeurons().get(i).getOutput());
            }
        }
        if (predictType.equals("degression") && predicts.isEmpty()) {
            errorListO.add(predict - output.getNeurons().get(0).getOutput());
        } else if (predictType.equals("degression") && predicts.size() > 0) {
            errorListO.add(getPredictValue(predicts) - output.getNeurons().get(0).getOutput());
        }

        /*
        Formação do arraylist de derivadas da sigmoid da saída (d(S))
         */
        ArrayList<Double> errorListD_O = new ArrayList<>();
        for (int i = 0; i < output.getNeuronsCount(); i++) {
            errorListD_O.add(i, FunctionActivation.sigmoidDer(output.getNeurons().get(i).getOutput()));
        }
        /*
        Gradiente - multiplicação da errorListD com errorList e formar um arraylist gradiente (E o d(S))
         */
        ArrayList<Double> gradiente_O = CalculatorHelper.multplyListByListAccordingToNumberOfNeurons(output, errorListO, errorListD_O);
        /*
        formção do arraylist gradiente * learning rage (E o d(S) * lr)
         */
        ArrayList<Double> gradiente_lr_O = CalculatorHelper.scaleListAccordingToNumberOfNeurons(output, gradiente_O, this.learningRate);

        /*
        ajuste do bias output
         */
//        for (int i = 0; i < output.getNeuronsCount(); i++) {
//            bias_o.set(i, bias_o.get(i) + gradiente_lr_O.get(i));
//        }
        bias_o = CalculatorHelper.addListByListAccordingToNumberOfNeurons(output, bias_o, gradiente_lr_O); //Todo Retestar pois esta esquisito
        this.deltaOW = new ArrayList<>();
        int auxConnectionsHidden = 0;
        while (auxConnectionsHidden < hidden.getNeuronsCount()) {
            for (int i = 0; i < hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().size(); i++) {
                deltaOW.add(i, deltaWeigthCalc2(gradiente_lr_O.get(i), hidden.getNeurons().get(auxConnectionsHidden).getNetInput()));
            }
            auxConnectionsHidden++;
        }
        auxConnectionsHidden = 0;
        while (auxConnectionsHidden < hidden.getNeuronsCount()) {
            for (int i = 0; i < hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().size(); i++) {
                hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().setValue(newWeightCalc(hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().getValue(), deltaOW.get(auxConnectionsHidden)));
            }
            auxConnectionsHidden++;
        }
    }

    public void backpropagationHiddenToAny() {
        //entrada para a oculta
        /*
        Formação do arraylist do calculo de cada erro
         */
        ArrayList<Double> errorListH = new ArrayList<>();
        int auxConnectionsHidden = 0;
        double valueHiddenError = 0;
        while (auxConnectionsHidden < hidden.getNeuronsCount()) {
            for (int i = 0; i < hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().size(); i++) {
                valueHiddenError += hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().getValue() * errorListO.get(i);
            }
//            System.out.println("Valor do valueHiddenError: " + valueHiddenError);
            errorListH.add(auxConnectionsHidden, valueHiddenError);
            auxConnectionsHidden++;
        }

        /*
        Formação do arraylist de derivadas da sigmoid da saída
         */
        ArrayList<Double> errorListHiddenD = new ArrayList<>();
        for (int i = 0; i < hidden.getNeuronsCount(); i++) {
            errorListHiddenD.add(i, FunctionActivation.sigmoidDer(errorListH.get(i)));
        }
        /*
        Gradiente - multiplicação da errorListD com errorList e formar um arraylist gradiente
         */
        ArrayList<Double> gradiente_H = CalculatorHelper.multplyListByListAccordingToNumberOfNeurons(hidden, errorListH, errorListHiddenD);
        /*
        formção do arraylist gradiente * learning rage
         */
        ArrayList<Double> gradiente_lr_H = CalculatorHelper.scaleListAccordingToNumberOfNeurons(hidden, gradiente_H, this.learningRate);
        /*
        ajuste do bias hidden
         */
//        for (int i = 0; i < hidden.getNeuronsCount(); i++) {
//            bias_h.set(i, bias_h.get(i) + gradiente_lr_H.get(i));
//        }
        bias_h = CalculatorHelper.addListByListAccordingToNumberOfNeurons(hidden, bias_h, gradiente_lr_H); //Todo Retestar pois esta esquisito
        this.deltaHW = new ArrayList<>();
        int auxConnectionsInput = 0;
        while (auxConnectionsInput < input.getNeuronsCount()) {
            for (int i = 0; i < input.getNeurons().get(auxConnectionsInput).getInputConnections().size(); i++) {
                deltaHW.add(i, deltaWeigthCalc2(gradiente_lr_H.get(i), input.getNeurons().get(auxConnectionsInput).getNetInput()));
//                System.out.println("Valores inseridos dentro da lista deltaHW: " + deltaHW.get(i));
            }
            auxConnectionsInput++;
        }

        auxConnectionsInput = 0;
        while (auxConnectionsInput < input.getNeuronsCount()) {
//            System.out.println("Calculando novos pesos da oculta até a entrada...");
            for (int i = 0; i < input.getNeurons().get(auxConnectionsInput).getInputConnections().size(); i++) {
                input.getNeurons().get(auxConnectionsInput).getInputConnections().get(i).getWeight().setValue(newWeightCalc(input.getNeurons().get(auxConnectionsInput).getInputConnections().get(i).getWeight().getValue(), deltaHW.get(auxConnectionsInput)));
//                System.out.println("Valores dos novos pesos: " + input.getNeurons().get(auxConnectionsInput).getInputConnections().get(i).getWeight().getValue());
            }
//            System.out.println("auxConnectionsInput: " + auxConnectionsInput);
            auxConnectionsInput++;
        }
    }

    public double errorCalc(double t, double s) {
//        return Math.floor((t - s) * 1000) / 1000;
        return t - s;
    }

    public double deltaBiasCalc(double e, double lr) {
//        return Math.floor((e * lr) * 1000) / 1000;
        return e * lr;
    }

    public double deltaWeigthCalc(double e, double lr, double n) {
//        return Math.floor((e * lr * input) * 1000) / 1000;
        //Todo realizar derivada da output
//        double dS = FunctionActivation.sigmoidDer(s);
        return e * lr * n;
    }

    public double deltaWeigthCalc2(double g, double n) {
        return g * n;
    }

    public double newBiasCalc(double b, double deltaB) {
        return Math.floor((deltaB + b) * 1000) / 1000;
//        return deltaB + b;
    }

    public double newWeightCalc(double w, double deltaW) {
        return Math.floor((deltaW + w) * 1000) / 1000;
//        return deltaW + w;
    }

    /*GETTERS E SETTERS*/
    public FunctionActivationData getFunctionActivation() {
        return functionActivation;
    }

    @Override
    public void setFunctionActivation(FunctionActivationData functionActivation) {
        this.functionActivation = functionActivation;
    }


    public ArrayList<Double> getOutputs() {
        return outputs;
    }

    public void setOutputs(ArrayList<Double> outputs) {
        this.outputs = outputs;
    }

    public double getPredict() {
        return predict;
    }

    private void setPredict(double predict) {
        this.predict = predict;
    }

    public void setNumberOfTrainings(int numberOfTrainings) {
        this.numberOfTrainings = numberOfTrainings;
    }

    @Override
    public double[] getInput(double[] in, double max, double min) {
//        System.out.println("Max: " + max + "Min: " + min);
        if (max != min) {
            double[] result = new double[in.length];
            for (int i = 0; i < result.length; i++) {
                result[i] = in[i] / (max - min);
//                System.out.println(result[i]);
            }
            return result;
        } else {
            return in;
        }
    }
}

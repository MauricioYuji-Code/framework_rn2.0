package network;

import core.*;
import Help.Helper;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Perceptron extends NeuralNetwork implements Serializable {

    //Objetos
    private Layer input;
    private Layer output;
    //Variáveis de entrada
    private double bias = 0;
    private double predict = 0;
    private double learningRate;
    private ArrayList<double[]> samples;
    private ArrayList<Double> inputsValues;
    //Variáveis de saida
    private double error;
    private ArrayList<Double> deltaW;
    private double deltaB;
    //Variáveis auxiliares
    private int samplesCountAux = 1;
    private int epoch = 1;
    private int trainingCount = 0;
    private int sampleCount = 0;
    //Feedfoward
    private ArrayList<double[]> inputsValuesReport;
    private ArrayList<Double> weightsInputValue;
    private double sumValue;
    private double outputValue;
    private boolean predictStatus;
    private double functionActivationResult;
    //Backpropagation
    private double errorValue;
    private ArrayList<Double> deltaWeightsValues;
    private ArrayList<Double> newWeightsValues;
    private double deltaBias;
    private double newBias;
    //Var for report
    private ArrayList<Double> initWeightsValues;
    private ArrayList<Double> initWeightsValuesReport;
    private String typeName = "Perceptron";
    private int numberLayerInput;
    private int numberNeuronInput;
    private String typeLayerNameInput;
    private int numberLayerOutput;
    private int numberNeuronOutput;
    private String typeLayerNameOutput;
    private FunctionActivationData functionActivation;
    private int samplePosition;
    //Assistants for report
    private ArrayList<Perceptron> reportFeedfoward = new ArrayList<>();
    Map<Integer, Perceptron> reportBackpropagation = new HashMap<>();
    private int round = 0;
    private ArrayList<Double> listInputData;
    private int nTraining = 0;

    public Perceptron() {

    }

    public Perceptron(double learningRate, double predict, double bias) {
        this.learningRate = learningRate;
        this.predict = predict;
        this.bias = bias;
    }

    public double errorCalc(double t, double s) {
        return t - s;
    }

    public double deltaWeigthCalc(double e, double lr, double input) {
        return e * lr * input;
    }

    public double deltaBiasCalc(double e, double lr) {
        return e * lr;
    }

    public double newWeightCalc(double w, double deltaW) {
        return deltaW + w;
    }

    public double newBiasCalc(double b, double deltaB) {
        return deltaB + b;
    }

    @Override
    public void setStructure(Type type, int nLayer, int nNeuron) {
        String aux = type.getTypeName();
        if (aux.equals("HIDDEN")) {
            System.out.println("Isso não é um perceptron!!");
            System.exit(0);
        }

        if (nLayer <= 0 || nNeuron <= 0) {
            System.out.println("Ops, algo errado no numero de camadas/neurônios");
            System.exit(0);
        }

        switch (aux) {
            case "INPUT":
                Layer input = new Layer(nNeuron);
                this.input = input;
                this.numberLayerInput = nLayer;
                this.numberNeuronInput = nNeuron;
                this.typeLayerNameInput = type.getTypeName();
                break;
            case "OUTPUT":
                Layer output = new Layer(nNeuron);
                this.output = output;
                this.numberLayerOutput = nLayer;
                this.numberNeuronOutput = nNeuron;
                this.typeLayerNameOutput = type.getTypeName();
                break;
            default:
                System.exit(0);
        }


    }

    @Override
    public void setInputValues(ArrayList inputValues) {
        this.samples = inputValues;
        this.inputsValues = inputValues;
        this.listInputData = new ArrayList<>();
        System.out.println("Lista de amostras para o treinamento");
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).setInput(samples.get(sampleCount)[i]);
            System.out.println(input.getNeurons().get(i).getNetInput());
            listInputData.add(i, input.getNeurons().get(i).getNetInput());
        }
    }

    public void setData(double[] data) {
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).setInput(data[i]);
            System.out.println(input.getNeurons().get(i).getNetInput());
        }
    }

    @Override
    public void connectNeuronIncludingWeigth(double weigthValue) {
        this.initWeightsValues = new ArrayList<>();
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            initWeightsValues.add(i, weigthValue);
            input.getNeurons().get(i).addInputConnection(output.getNeurons().get(0), weigthValue);
        }
    }

    public void checkNextSamples() {
        Helper.drawLine();
        System.out.println("Checagem das próximas amostras");
        if (samplesCountAux != samples.size()) {
            trainWiththeNextSamples();
        } else {
            generateNewEpoch();
        }
    }

    public void trainWiththeNextSamples() {
        System.out.println("Proximas amostras encontradas");
        Helper.drawLine();
        System.out.println("Começando o treinamento!");
        sampleCount++;
        setInputValues(samples);
        samplesCountAux++;
        training();
    }

    public void generateNewEpoch() {
        this.epoch++;
        if (trainingCount > 0) {
            this.sampleCount = 0;
            this.samplesCountAux = 0;
            this.round = 0;
            setInputValues(samples);
        }
        while (trainingCount > 0) {
            samplesCountAux++;
            this.trainingCount = 0;
            training();
        }
        this.sampleCount = 0;
    }

    public void start() {
        System.out.println("Start Perceptron!!");
        selectFunctionActivation();
        Helper.drawLine();
        if (output.getNeurons().get(0).getOutput() == predict) {
            Helper.drawLine();
            System.out.println("Valor da saída: " + output.getNeurons().get(0).getOutput() + "Valor da predição: " + predict);
            System.out.println("Valores conferem");
        } else {
            Helper.drawLine();
            System.out.println("Valor da saída: " + output.getNeurons().get(0).getOutput() + "Valor da predição: " + predict);
            System.out.println("Valores não conferem");
        }

    }

    public void training() {
        Helper.drawLine();
        System.out.println("Começando treinamento do Perceptron");
        selectFunctionActivation();
        reportFeedfoward();
        round++;
        while (output.getNeurons().get(0).getOutput() != predict) {
            Helper.drawLine();
            predictStatus = false;
            System.out.println("A rede precisa de treinamento, resultado não corresponde com o esperado");
            backPropagation();
            selectFunctionActivation();
            activateCounters();
            defineStatus();
            reportBackpropagation();
        }
        Helper.drawLine();
        System.out.println("Rede treinada!");
        checkNextSamples();
    }

    public void activateCounters() {
        trainingCount++;
        nTraining++;
        round++;
    }

    public void defineStatus() {
        if (output.getNeurons().get(0).getOutput() == predict) {
            predictStatus = true;
        } else {
            predictStatus = false;
        }
    }

    public void selectFunctionActivation() {
        if (getFunctionActivaion().name().equals("DEGRAU")) {
            output.getNeurons().get(0).setOutput(FunctionActivation.degrau(sum()));
            functionActivationResult = FunctionActivation.degrau(sum());
        } else if (getFunctionActivaion().name().equals("SIGMOID")) {
            output.getNeurons().get(0).setOutput(FunctionActivation.sigmoid(sum()));
            functionActivationResult = FunctionActivation.degrau(sum());
        }
    }

    public double sum() {
        double aux = 0;
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            aux += (input.getNeurons().get(i).getNetInput() * input.getNeurons().get(i).getInputConnections().get(0).getWeight().getValue());
        }
        aux = aux + bias;
        return aux;
    }

    public void backPropagation() {
        this.error = errorCalc(predict, output.getNeurons().get(0).getOutput());
        this.deltaW = new ArrayList<>();
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            deltaW.add(i, deltaWeigthCalc(error, learningRate, input.getNeurons().get(i).getNetInput()));
        }
        this.deltaB = deltaBiasCalc(error, learningRate);
        this.newWeightsValues = new ArrayList<>();
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).getInputConnections().get(0).getWeight().setValue(newWeightCalc(input.getNeurons().get(i).getInputConnections().get(0).getWeight().getValue(), deltaW.get(i)));
            this.newWeightsValues.add(i, newWeightCalc(input.getNeurons().get(i).getInputConnections().get(0).getWeight().getValue(), deltaW.get(i))); //report
        }
        this.initWeightsValues = newWeightsValues;
        this.bias = newBiasCalc(bias, deltaB);
    }

    public void reportFeedfoward() {
        Perceptron pFeedfoward = new Perceptron();
        pFeedfoward.setSumValue(sum());
        pFeedfoward.setOutputValue(output.getNeurons().get(0).getOutput());
        pFeedfoward.setNewBias(bias);
        pFeedfoward.setEpoch(epoch);
        pFeedfoward.setTrainingCount(trainingCount);
        pFeedfoward.setRound(round);
        pFeedfoward.setInputsValuesReport(samples);
        pFeedfoward.setInitWeightsValuesReport(initWeightsValues);
        pFeedfoward.setFunctionActivation(getFunctionActivaion());
        pFeedfoward.setSamplePosition(sampleCount);
        pFeedfoward.setPredictStatus(predictStatus);
        pFeedfoward.setLearningRateValue(learningRate);
        pFeedfoward.setPredictValue(predict);
        pFeedfoward.setBiasValue(bias);
        pFeedfoward.setFunctionActivationResult(functionActivationResult);
        pFeedfoward.setStructureInputReport(numberNeuronInput, numberLayerInput, typeLayerNameInput);
        pFeedfoward.setStructureOutputReport(numberNeuronOutput, numberLayerOutput, typeLayerNameOutput);
        pFeedfoward.setListInputData(listInputData);
        pFeedfoward.setnTraining(nTraining);
        reportFeedfoward.add(pFeedfoward);
    }

    public void reportBackpropagation() {
        Perceptron pBackPropagation = new Perceptron();
        pBackPropagation.setSumValue(sum());
        pBackPropagation.setOutputValue(output.getNeurons().get(0).getOutput());
        pBackPropagation.setErrorValue(error);
        pBackPropagation.setDeltaWeightsValues(deltaW);
        pBackPropagation.setNewWeightsValues(newWeightsValues);
        pBackPropagation.setDeltaBias(deltaB);
        pBackPropagation.setNewBias(bias);
        pBackPropagation.setEpoch(epoch);
        pBackPropagation.setTrainingCount(trainingCount);
        pBackPropagation.setRound(round);
        pBackPropagation.setInputsValuesReport(samples);
        pBackPropagation.setInitWeightsValuesReport(initWeightsValues);
        pBackPropagation.setFunctionActivation(getFunctionActivaion());
        pBackPropagation.setSamplePosition(sampleCount);
        pBackPropagation.setPredictStatus(predictStatus);
        pBackPropagation.setFunctionActivationResult(functionActivationResult);
        pBackPropagation.setListInputData(listInputData);
        pBackPropagation.setnTraining(nTraining);
        reportBackpropagation.put((reportFeedfoward.size() - 1), pBackPropagation);
    }

//    *******Getter's and Setter's

    public ArrayList<double[]> getSamplesValues() {
        return samples;
    }

    private void setSamplesValues(ArrayList<double[]> samplesValues) {
        this.samples = samplesValues;
    }

    public ArrayList<Double> getInitWeightsValues() {
        return initWeightsValues;
    }

    private void setInitWeightsValues(ArrayList<Double> initWeightsValues) {
        this.initWeightsValues = initWeightsValues;
    }

    public double getLearningRateValue() {
        return learningRate;
    }

    private void setLearningRateValue(double learningRateValue) {
        this.learningRate = learningRateValue;
    }

    public double getPredictValue() {
        return predict;
    }

    public void setPredictValue(double predictValue) {
        this.predict = predictValue;
    }

    public double getBiasValue() {
        return bias;
    }

    private void setBiasValue(double biasValue) {
        this.bias = biasValue;
    }

    public String getTypeName() {
        return typeName;
    }

//    private void setTypeName(String typeName) {
//        this.typeName = typeName;
//    }

    public ArrayList<Double> getInputsValues() {
        return inputsValues;
    }

    private void setInputsValues(ArrayList<Double> inputsValues) {
        this.inputsValues = inputsValues;
    }

    public ArrayList<Double> getWeightsInputValue() {
        return weightsInputValue;
    }

    private void setWeightsInputValue(ArrayList<Double> weightsInputValue) {
        this.weightsInputValue = weightsInputValue;
    }

    public double getSumValue() {
        return sumValue;
    }

    private void setSumValue(double sumValue) {
        this.sumValue = sumValue;
    }

    public double getOutputValue() {
        return outputValue;
    }

    private void setOutputValue(double outputValue) {
        this.outputValue = outputValue;
    }

    public boolean getPredictStatus() {
        return predictStatus;
    }

    private void setPredictStatus(boolean predictStatus) {
        this.predictStatus = predictStatus;
    }

    public double getErrorValue() {
        return errorValue;
    }

    private void setErrorValue(double errorValue) {
        this.errorValue = errorValue;
    }

    public ArrayList<Double> getDeltaWeightsValues() {
        return deltaWeightsValues;
    }

    private void setDeltaWeightsValues(ArrayList<Double> deltaWeightsValues) {
        this.deltaWeightsValues = deltaWeightsValues;
    }

    public ArrayList<Double> getNewWeightsValues() {
        return newWeightsValues;
    }

    private void setNewWeightsValues(ArrayList<Double> newWeightsValues) {
        this.newWeightsValues = newWeightsValues;
    }

    public double getDeltaBias() {
        return deltaBias;
    }

    private void setDeltaBias(double deltaBias) {
        this.deltaBias = deltaBias;
    }

    public double getNewBias() {
        return newBias;
    }

    private void setNewBias(double newBias) {
        this.newBias = newBias;
    }

    public FunctionActivationData getFunctionActivaion() {
        return functionActivation;
    }

    @Override
    public void setFunctionActivation(FunctionActivationData functionActivation) {
        this.functionActivation = functionActivation;
    }

    public ArrayList<Perceptron> getReportFeedfoward() {
        return reportFeedfoward;
    }

    public int getEpoch() {
        return epoch;
    }

    private void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    private void setTrainingCount(int trainingCount) {
        this.trainingCount = trainingCount;
    }

    public int getRound() {
        return round;
    }

    private void setRound(int round) {
        this.round = round;
    }

    public ArrayList<double[]> getInputsValuesReport() {
        return inputsValuesReport;
    }

    private void setInputsValuesReport(ArrayList<double[]> inputsValuesReport) {
        this.inputsValuesReport = inputsValuesReport;
    }

    public ArrayList<Double> getInitWeightsValuesReport() {
        return initWeightsValuesReport;
    }

    private void setInitWeightsValuesReport(ArrayList<Double> initWeightsValuesReport) {
        this.initWeightsValuesReport = initWeightsValuesReport;
    }

    public int getSamplePosition() {
        return samplePosition;
    }

    private void setSamplePosition(int samplePosition) {
        this.samplePosition = samplePosition;
    }

    public double getFunctionActivationResult() {
        return functionActivationResult;
    }

    private void setFunctionActivationResult(double functionActivationResult) {
        this.functionActivationResult = functionActivationResult;
    }

    private void setStructureInputReport(int numberNeuron, int numberLayer, String typeLayerName) {
        this.numberNeuronInput = numberNeuron;
        this.numberLayerInput = numberLayer;
        this.typeLayerNameInput = typeLayerName;
    }

    private void setStructureOutputReport(int numberNeuron, int numberLayer, String typeLayerName) {
        this.numberNeuronOutput = numberNeuron;
        this.numberLayerOutput = numberLayer;
        this.typeLayerNameOutput = typeLayerName;
    }

    public String getStructureInputReport() {
        return typeLayerNameInput + " " + (numberLayerInput) + " " + (numberNeuronInput);
    }

    public String getStructureOutputReport() {
        return typeLayerNameOutput + " " + (numberLayerOutput) + " " + (numberNeuronOutput);
    }

    public ArrayList<Double> getListInputData() {
        return listInputData;
    }

    private void setListInputData(ArrayList<Double> listInputData) {
        this.listInputData = listInputData;
    }

    public int getnTraining() {
        return nTraining;
    }

    private void setnTraining(int nTraining) {
        this.nTraining = nTraining;
    }

    public Map<Integer, Perceptron> getReportBackpropagation() {
        return reportBackpropagation;
    }

}
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
    private int samplesCount = 1;
    private int epoch = 1;
    private int trainingCount = 0;
    private int samplePositionList = 0; //samplePositionOf
    private int round = 0;
    private int nTraining = 0;
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
    private Map<Integer, Perceptron> reportBackpropagation = new HashMap<>();
    private ArrayList<Double> listInputData;

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
            input.getNeurons().get(i).setInput(samples.get(samplePositionList)[i]);
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
        if (samplesCount != samples.size()) {
            trainWiththeNextSamples();
        } else {
            generateNewEpoch();
        }
    }

    public void trainWiththeNextSamples() {
        Helper.drawLine();
        System.out.println("Começando o treinamento!");
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
            setInputValues(samples);
        }
        while (trainingCount > 0) {
            samplesCount++;
            this.trainingCount = 0;
            training();
        }
        this.samplePositionList = 0;
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
//            reportFeedfoward();
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
            this.functionActivationResult = FunctionActivation.degrau(sum());
            this.sumValue = sum();
        } else if (getFunctionActivaion().name().equals("SIGMOID")) {
            output.getNeurons().get(0).setOutput(FunctionActivation.sigmoid(sum())); //Feedfoward
            this.functionActivationResult = FunctionActivation.degrau(sum());
            this.sumValue = sum();
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
        pFeedfoward.setSumValue(sumValue);
        pFeedfoward.setOutputValue(output.getNeurons().get(0).getOutput());
        pFeedfoward.setNewBias(bias);
        pFeedfoward.setEpoch(epoch);
        pFeedfoward.setTrainingCount(trainingCount);
        pFeedfoward.setRound(round);
        pFeedfoward.setInputsValuesReport(samples);
        pFeedfoward.setInitWeightsValuesReport(initWeightsValues);
        pFeedfoward.setFunctionActivation(getFunctionActivaion());
        pFeedfoward.setSamplePosition(samplePositionList);
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
        pBackPropagation.setSamplePosition(samplePositionList);
        pBackPropagation.setPredictStatus(predictStatus);
        pBackPropagation.setFunctionActivationResult(functionActivationResult);
        pBackPropagation.setListInputData(listInputData);
        pBackPropagation.setnTraining(nTraining);
        reportBackpropagation.put((reportFeedfoward.size() - 1), pBackPropagation);
    }

    /*************************GETTERS*************************/

    public ArrayList<double[]> getSamplesValues() {
        return samples;
    }

    public ArrayList<Double> getInitWeightsValues() {
        return initWeightsValues;
    }

    public double getLearningRateValue() {
        return learningRate;
    }

    public double getPredictValue() {
        return predict;
    }

    public double getBiasValue() {
        return bias;
    }

    public String getTypeName() {
        return typeName;
    }

    public ArrayList<Double> getInputsValues() {
        return inputsValues;
    }

    public ArrayList<Double> getWeightsInputValue() {
        return weightsInputValue;
    }

    public double getSumValue() {
        return sumValue;
    }

    public double getOutputValue() {
        return outputValue;
    }

    public boolean getPredictStatus() {
        return predictStatus;
    }

    public double getErrorValue() {
        return errorValue;
    }

    public double getDeltaBias() {
        return deltaBias;
    }

    public ArrayList<Double> getDeltaWeightsValues() {
        return deltaWeightsValues;
    }

    public ArrayList<Double> getNewWeightsValues() {
        return newWeightsValues;
    }

    public double getNewBias() {
        return newBias;
    }

    public FunctionActivationData getFunctionActivaion() {
        return functionActivation;
    }

    public ArrayList<Perceptron> getReportFeedfoward() {
        return reportFeedfoward;
    }

    public int getEpoch() {
        return epoch;
    }

    public int getSamplePosition() {
        return samplePosition;
    }

    public ArrayList<double[]> getInputsValuesReport() {
        return inputsValuesReport;
    }

    public int getRound() {
        return round;
    }

    public ArrayList<Double> getInitWeightsValuesReport() {
        return initWeightsValuesReport;
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

    public double getFunctionActivationResult() {
        return functionActivationResult;
    }

    public int getnTraining() {
        return nTraining;
    }

    public Map<Integer, Perceptron> getReportBackpropagation() {
        return reportBackpropagation;
    }

    /*************************SETTERS*************************/

    private void setSamplesValues(ArrayList<double[]> samplesValues) {
        this.samples = samplesValues;
    }

    private void setInitWeightsValues(ArrayList<Double> initWeightsValues) {
        this.initWeightsValues = initWeightsValues;
    }

    private void setLearningRateValue(double learningRateValue) {
        this.learningRate = learningRateValue;
    }

    public void setPredictValue(double predictValue) {
        this.predict = predictValue;
    }

    private void setBiasValue(double biasValue) {
        this.bias = biasValue;
    }

//    private void setTypeName(String typeName) {
//        this.typeName = typeName;
//    }

    private void setInputsValues(ArrayList<Double> inputsValues) {
        this.inputsValues = inputsValues;
    }

    private void setWeightsInputValue(ArrayList<Double> weightsInputValue) {
        this.weightsInputValue = weightsInputValue;
    }

    private void setSumValue(double sumValue) {
        this.sumValue = sumValue;
    }

    private void setOutputValue(double outputValue) {
        this.outputValue = outputValue;
    }

    private void setPredictStatus(boolean predictStatus) {
        this.predictStatus = predictStatus;
    }

    private void setErrorValue(double errorValue) {
        this.errorValue = errorValue;
    }

    private void setDeltaWeightsValues(ArrayList<Double> deltaWeightsValues) {
        this.deltaWeightsValues = deltaWeightsValues;
    }

    private void setNewWeightsValues(ArrayList<Double> newWeightsValues) {
        this.newWeightsValues = newWeightsValues;
    }

    private void setDeltaBias(double deltaBias) {
        this.deltaBias = deltaBias;
    }

    private void setNewBias(double newBias) {
        this.newBias = newBias;
    }

    @Override
    public void setFunctionActivation(FunctionActivationData functionActivation) {
        this.functionActivation = functionActivation;
    }

    private void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    private void setTrainingCount(int trainingCount) {
        this.trainingCount = trainingCount;
    }

    private void setRound(int round) {
        this.round = round;
    }

    private void setInputsValuesReport(ArrayList<double[]> inputsValuesReport) {
        this.inputsValuesReport = inputsValuesReport;
    }

    private void setInitWeightsValuesReport(ArrayList<Double> initWeightsValuesReport) {
        this.initWeightsValuesReport = initWeightsValuesReport;
    }

    private void setSamplePosition(int samplePosition) {
        this.samplePosition = samplePosition;
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

    private void setListInputData(ArrayList<Double> listInputData) {
        this.listInputData = listInputData;
    }

    private void setnTraining(int nTraining) {
        this.nTraining = nTraining;
    }

}
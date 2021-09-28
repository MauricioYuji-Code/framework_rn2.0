package network;

import core.*;
import utils.FunctionActivationData;
import utils.Helper;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.ArrayList;

public class Perceptron extends NeuralNetwork implements Serializable {
    private double bias = 0;
    private double predict = 0;
    private double error;
    private ArrayList<Double> deltaW;
    private double deltaB;
    private double learningRate = 0.1;
    private Layer input;
    private Layer output;
    private int sampleCount = 0;
    private ArrayList<double[]> samples;
    //Variaveis auxiliares
    private int samplesCountAux = 1;
    private int epoch = 1;
    private int trainingCount = 0;

    //Teste

    //Dados iniciais
    private ArrayList<Double> initWeightsValues;
    private String typeName = "Perceptron";
    private int numberLayer;
    private int numberNeuron;
    private FunctionActivationData functionActivation;
    //Dados de execução do treinamento
    //Feedfoward
    private ArrayList<Double> inputsValues;
    private ArrayList<Double> weightsInputValue;
    private double sumValue;
    private double outputValue;
    private boolean predictStatus;
    //Backpropagation
    private double errorValue; //ok
    private ArrayList<Double> deltaWeightsValues; //ok
    private ArrayList<Double> newWeightsValues; //ok
    private double deltaBias; //ok
    private double newBias; //ok
    //Report
    private ArrayList<Perceptron> reports = new ArrayList<>();
    private int round = 0;

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
        this.numberLayer = nLayer;
        this.numberNeuron = nNeuron;
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
//                System.out.println("Camada sendo estruturada " + type);
                Layer input = new Layer(nNeuron);
                this.input = input;
                break;
            case "OUTPUT":
//                System.out.println("Camada sendo estruturada " + type);
                Layer output = new Layer(nNeuron);
                this.output = output;
                break;
            default:
//                System.out.println("Você não utilizou nem input ou output como tipo da camada");
                System.exit(0);
        }


    }

    @Override
    public void setInputValues(ArrayList inputValues) {
        this.samples = inputValues;
        this.inputsValues = inputValues;
        System.out.println("Lista de amostras para o treinamento");
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).setInput(samples.get(sampleCount)[i]);
            System.out.println(input.getNeurons().get(i).getNetInput());
        }
    }

    public void setData(double[] data){
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).setInput(data[i]);
            System.out.println(input.getNeurons().get(i).getNetInput());
        }
    }

    //Feedfoward
    @Override
    public void connectNeuronIncludingWeigth(double weigthValue) {
        this.initWeightsValues = new ArrayList<>();
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            initWeightsValues.add(i, weigthValue);
//            System.out.println(input.getNeurons().get(i).getNetInput());
            input.getNeurons().get(i).addInputConnection(output.getNeurons().get(0), weigthValue);
//            System.out.println("O neurônio " + input.getNeurons().get(i).getNetInput() + " tem conexão? " + input.getNeurons().get(i).hasInputConnections());
        }
    }

    public void checkNextSamples() {
        Helper.drawLine();
        System.out.println("Checagem das próximas amostras");
        if (samplesCountAux != samples.size()) {
            System.out.println("Proximas amostras encontradas");
            Helper.drawLine();
            System.out.println("Começando o treinamento!");
            sampleCount++;
            setInputValues(samples);
            samplesCountAux++;
            training();
        } else {
            Helper.drawLine();
            System.out.println("Iniando a próxima época");
            this.epoch++;
            if (trainingCount > 0) {
                this.sampleCount = 0;
                this.samplesCountAux = 0;
                this.round = 0;
                setInputValues(samples);
            }
            while (trainingCount > 0) {
//                System.out.println("Sample Count: " + sampleCount);
//                System.out.println("Training Count: " + trainingCount);
                samplesCountAux++;
                this.trainingCount = 0;
                training();
            }
            this.sampleCount = 0;
        }
    }

    public void start() {
        System.out.println("Start Perceptron!!");
        selectFunctionActivation();
//        System.out.println("Valor do neurônio de saída: " + output.getNeurons().get(0).getOutput());
        Helper.drawLine();
        if (output.getNeurons().get(0).getOutput() == predict) {
            Helper.drawLine();
            System.out.println("Valor da saída: " + output.getNeurons().get(0).getOutput() +  "Valor da predição: " + predict);
            System.out.println("Valores conferem");
        } else {
            Helper.drawLine();
            System.out.println("Valor da saída: " + output.getNeurons().get(0).getOutput() +  "Valor da predição: " + predict);
            System.out.println("Valores não conferem");
        }

    }

    public void training() {
        Helper.drawLine();
        System.out.println("Começando treinamento do Perceptron");
        selectFunctionActivation();
//        System.out.println("Valor do neurônio de saída: " + output.getNeurons().get(0).getOutput());
        //Primeira iteração
        round++;
        reportStart();
        while (output.getNeurons().get(0).getOutput() != predict) {
            Helper.drawLine();
            System.out.println("Resultado final da saída: " + output.getNeurons().get(0).getOutput() + " Valor esperado: " + predict);
            System.out.println("A rede precisa de treinamento, resultado não corresponde com o esperado");
            System.out.println("recomeçando o treinamento");
            backPropagation();
            selectFunctionActivation();
//            System.out.println("Valor do neurônio de saída: " + output.getNeurons().get(0).getOutput());
            predictStatus = false;
            trainingCount++;
            round++;
            reportTraining();
        }
        predictStatus = true;
        Helper.drawLine();
        System.out.println("Rede treinada! \nResultado final da saída: " + output.getNeurons().get(0).getOutput() + " Valor esperado: " + predict);
        checkNextSamples();
    }

    public void selectFunctionActivation(){
        if(getFunctionActivaion().name().equals("DEGRAU")) {
            output.getNeurons().get(0).setOutput(FunctionActivation.degrau(sum()));
        }else if(getFunctionActivaion().name().equals("SIGMOID")){
            output.getNeurons().get(0).setOutput(FunctionActivation.sigmoid(sum()));
        }
    }

    public double sum() {
//        System.out.println("Realizando a somatória...");
        double aux = 0;
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            aux += (input.getNeurons().get(i).getNetInput() * input.getNeurons().get(i).getInputConnections().get(0).getWeight().getValue());
        }
        aux = aux + bias;
//        System.out.println("Resultado da somatória: " + aux);
        return aux;
    }

    public void backPropagation() {
        this.error = errorCalc(predict, output.getNeurons().get(0).getOutput());
//        System.out.println("Valor do erro..." + error);
//        System.out.println("Cálculo variação do peso...");
        this.deltaW = new ArrayList<>();
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            deltaW.add(i, deltaWeigthCalc(error, learningRate, input.getNeurons().get(i).getNetInput()));
//            System.out.println("Valores inseridos dentro da lista deltaW: " + deltaW.get(i));
        }
        setDeltaWeightsValues(deltaW); //report
//        System.out.println("Cálculo variação do bias...");
        this.deltaB = deltaBiasCalc(error, learningRate);
        setDeltaBias(deltaB); //report
//        System.out.println("Criação do bias..." + deltaB);
//        System.out.println("Calculando novos pesos...");
        this.newWeightsValues = new ArrayList<>();
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).getInputConnections().get(0).getWeight().setValue(newWeightCalc(input.getNeurons().get(i).getInputConnections().get(0).getWeight().getValue(), deltaW.get(i)));
//            System.out.println("Valores dos novos pesos: " + input.getNeurons().get(i).getInputConnections().get(0).getWeight().getValue());
            this.newWeightsValues.add(i, newWeightCalc(input.getNeurons().get(i).getInputConnections().get(0).getWeight().getValue(), deltaW.get(i))); //report
        }
//        System.out.println("Novo bias...");
        this.bias = newBiasCalc(bias, deltaB);
        setNewBias(bias); //report
//        System.out.println("Valor do novo bias..." + bias);
    }

    public void reportStart() {
        Perceptron pStart = new Perceptron();
        pStart.setSumValue(sum());
        pStart.setOutputValue(output.getNeurons().get(0).getOutput());
        pStart.setErrorValue(error);
        pStart.setDeltaWeightsValues(deltaW);
        pStart.setNewWeightsValues(newWeightsValues);
        pStart.setDeltaBias(deltaB);
        pStart.setNewBias(bias);
        pStart.setEpoch(epoch);
        pStart.setTrainingCount(trainingCount);
        pStart.setRound(round);
//        System.out.println("Valor da rodada no Start: " + round);
//        System.out.println("valor da auxReportPosition: " + auxReportPosition);
        reports.add(pStart);
    }

    public void reportTraining() {
        Perceptron pTraining = new Perceptron();
        pTraining.setSumValue(sum());
        pTraining.setOutputValue(output.getNeurons().get(0).getOutput());
        pTraining.setErrorValue(error);
        pTraining.setDeltaWeightsValues(deltaW); //report
        pTraining.setNewWeightsValues(newWeightsValues);
        pTraining.setDeltaBias(deltaB);
        pTraining.setNewBias(bias);
        pTraining.setEpoch(epoch);
        pTraining.setTrainingCount(trainingCount);
        pTraining.setRound(round);
//        System.out.println("Valor da rodada no Training: " + round);
        //Demais iterações
//        auxReportPosition++;
//        System.out.println("valor da auxReportPosition: " + auxReportPosition);
        reports.add(pTraining);
    }

//    //Todo Jean - Desenvoler o front nessa função
//    @Override
//    public void report(ArrayList<Perceptron> r){
//
//
//    }

//    *******Getter's and Setter's

    public ArrayList<double[]> getSamplesValues() {
        return samples;
    }

    public void setSamplesValues(ArrayList<double[]> samplesValues) {
        this.samples = samplesValues;
    }

    public ArrayList<Double> getInitWeightsValues() {
        return initWeightsValues;
    }

    public void setInitWeightsValues(ArrayList<Double> initWeightsValues) {
        this.initWeightsValues = initWeightsValues;
    }

    public double getLearningRateValue() {
        return learningRate;
    }

    public void setLearningRateValue(double learningRateValue) {
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

    public void setBiasValue(double biasValue) {
        this.bias = biasValue;
    }

    public String getTypeName() {
        return typeName;
    }

    public void setTypeName(String typeName) {
        this.typeName = typeName;
    }

    public int getNumberLayer() {
        return numberLayer;
    }

    public void setNumberLayer(int numberLayer) {
        this.numberLayer = numberLayer;
    }

    public int getNumberneuron() {
        return numberNeuron;
    }

    public void setNumberneuron(int numberneuron) {
        this.numberNeuron = numberneuron;
    }

    public ArrayList<Double> getInputsValues() {
        return inputsValues;
    }

    public void setInputsValues(ArrayList<Double> inputsValues) {
        this.inputsValues = inputsValues;
    }

    public ArrayList<Double> getWeightsInputValue() {
        return weightsInputValue;
    }

    public void setWeightsInputValue(ArrayList<Double> weightsInputValue) {
        this.weightsInputValue = weightsInputValue;
    }

    public double getSumValue() {
        return sumValue;
    }

    public void setSumValue(double sumValue) {
        this.sumValue = sumValue;
    }

    public double getOutputValue() {
        return outputValue;
    }

    public void setOutputValue(double outputValue) {
        this.outputValue = outputValue;
    }

    public boolean isPredictStatus() {
        return predictStatus;
    }

    public void setPredictStatus(boolean predictStatus) {
        this.predictStatus = predictStatus;
    }

    public double getErrorValue() {
        return errorValue;
    }

    public void setErrorValue(double errorValue) {
        this.errorValue = errorValue;
    }

    public ArrayList<Double> getDeltaWeightsValues() {
        return deltaWeightsValues;
    }

    public void setDeltaWeightsValues(ArrayList<Double> deltaWeightsValues) {
        this.deltaWeightsValues = deltaWeightsValues;
    }

    public ArrayList<Double> getNewWeightsValues() {
        return newWeightsValues;
    }

    public void setNewWeightsValues(ArrayList<Double> newWeightsValues) {
        this.newWeightsValues = newWeightsValues;
    }

    public double getDeltaBias() {
        return deltaBias;
    }

    public void setDeltaBias(double deltaBias) {
        this.deltaBias = deltaBias;
    }

    public double getNewBias() {
        return newBias;
    }

    public void setNewBias(double newBias) {
        this.newBias = newBias;
    }

    public FunctionActivationData getFunctionActivaion() {
        return functionActivation;
    }

    @Override
    public void setFunctionActivation(FunctionActivationData functionActivation) {
        this.functionActivation = functionActivation;
    }

    public ArrayList<Perceptron> getReports() {
        return reports;
    }

    public int getEpoch() {
        return epoch;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    public void setReports(ArrayList<Perceptron> reports) {
        this.reports = reports;
    }

    public int getTrainingCount() {
        return trainingCount;
    }

    public void setTrainingCount(int trainingCount) {
        this.trainingCount = trainingCount;
    }

    public int getRound() {
        return round;
    }

    public void setRound(int round) {
        this.round = round;
    }
}
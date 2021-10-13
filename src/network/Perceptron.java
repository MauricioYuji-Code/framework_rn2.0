package network;

import core.*;
import Help.Helper;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.ArrayList;

public class Perceptron extends NeuralNetwork implements Serializable {
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
    private Layer input;
    private Layer output;
    //Variáveis auxiliares
    private int samplesCountAux = 1;
    private int epoch = 1;
    private int trainingCount = 0;
    private int sampleCount = 0;
    //Variáveis do report
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
    //Report
    private ArrayList<Perceptron> reports = new ArrayList<>();
    private int round = 0;
    //    private Layer inputReport;
    private ArrayList<Double> inputReport;
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
//        this.numberLayer = nLayer;
//        this.numberNeuron = nNeuron;
//        this.typeLayerName = type.getTypeName();
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
                this.numberLayerInput = nLayer;
                this.numberNeuronInput = nNeuron;
                this.typeLayerNameInput = type.getTypeName();
                break;
            case "OUTPUT":
//                System.out.println("Camada sendo estruturada " + type);
                Layer output = new Layer(nNeuron);
                this.output = output;
                this.numberLayerOutput = nLayer;
                this.numberNeuronOutput = nNeuron;
                this.typeLayerNameOutput = type.getTypeName();
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
        inputReport = new ArrayList<>();
        System.out.println("Lista de amostras para o treinamento");
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).setInput(samples.get(sampleCount)[i]);
            System.out.println(input.getNeurons().get(i).getNetInput());
            inputReport.add(i, input.getNeurons().get(i).getNetInput());
        }
    }

    public void setData(double[] data) {
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
//            Helper.drawLine();
//            System.out.println("Iniando a próxima época");
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
//        System.out.println("Valor do neurônio de saída: " + output.getNeurons().get(0).getOutput());
        //Primeira iteração
        round++;
        reportStart();
        while (output.getNeurons().get(0).getOutput() != predict) {
            Helper.drawLine();
            predictStatus = false;
            System.out.println("Resultado final da saída: " + output.getNeurons().get(0).getOutput() + " Valor esperado: " + predict);
            System.out.println("A rede precisa de treinamento, resultado não corresponde com o esperado");
            System.out.println("recomeçando o treinamento");
            backPropagation();
            selectFunctionActivation();
//            System.out.println("Valor do neurônio de saída: " + output.getNeurons().get(0).getOutput());
            trainingCount++;
            nTraining++;
            round++;
            if (output.getNeurons().get(0).getOutput() == predict) {
                predictStatus = true;
            } else {
                predictStatus = false;
            }
            reportTraining();
        }
        Helper.drawLine();
        System.out.println("Rede treinada! \nResultado final da saída: " + output.getNeurons().get(0).getOutput() + " Valor esperado: " + predict);
        predictStatus = true;
        checkNextSamples();
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
//        System.out.println("Criação do bias..." + deltaB);
//        System.out.println("Calculando novos pesos...");
        this.newWeightsValues = new ArrayList<>();
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).getInputConnections().get(0).getWeight().setValue(newWeightCalc(input.getNeurons().get(i).getInputConnections().get(0).getWeight().getValue(), deltaW.get(i)));
            this.newWeightsValues.add(i, newWeightCalc(input.getNeurons().get(i).getInputConnections().get(0).getWeight().getValue(), deltaW.get(i))); //report
        }
        this.initWeightsValues = newWeightsValues;
//        System.out.println("teste dentro do perceptron: " + initWeightsValuesReport.get(1));
//        System.out.println("Novo bias...");
        this.bias = newBiasCalc(bias, deltaB);
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
//        pStart.setNumberneuron(numberNeuron);
        pStart.setInputsValuesReport(samples);
        pStart.setInitWeightsValuesReport(initWeightsValues);
        pStart.setFunctionActivation(getFunctionActivaion());
        pStart.setSamplePosition(sampleCount);
        pStart.setPredictStatus(predictStatus);
        pStart.setLearningRateValue(learningRate);
        pStart.setPredictValue(predict);
        pStart.setBiasValue(bias);
        pStart.setFunctionActivationResult(functionActivationResult);
        pStart.setStructureInputReport(numberNeuronInput, numberLayerInput, typeLayerNameInput);
        pStart.setStructureOutputReport(numberNeuronOutput, numberLayerOutput, typeLayerNameOutput);
        pStart.setInputReport(inputReport);
        pStart.setnTraining(nTraining);
        reports.add(pStart);
    }

    public void reportTraining() {
        Perceptron pTraining = new Perceptron();
        pTraining.setSumValue(sum());
        pTraining.setOutputValue(output.getNeurons().get(0).getOutput());
        pTraining.setErrorValue(error);
        pTraining.setDeltaWeightsValues(deltaW);
        pTraining.setNewWeightsValues(newWeightsValues);
        pTraining.setDeltaBias(deltaB);
        pTraining.setNewBias(bias);
        pTraining.setEpoch(epoch);
        pTraining.setTrainingCount(trainingCount);
        pTraining.setRound(round);
//        pTraining.setNumberneuron(numberNeuron);
        pTraining.setInputsValuesReport(samples);
        pTraining.setInitWeightsValuesReport(initWeightsValues);
        pTraining.setFunctionActivation(getFunctionActivaion());
        pTraining.setSamplePosition(sampleCount);
        pTraining.setPredictStatus(predictStatus);
        pTraining.setFunctionActivationResult(functionActivationResult);
        pTraining.setInputReport(inputReport);
        pTraining.setnTraining(nTraining);
        reports.add(pTraining);
    }

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

//    public int getNumberLayer() {
//        return numberLayer;
//    }
//
//    public void setNumberLayer(int numberLayer) {
//        this.numberLayer = numberLayer;
//    }
//
//    public int getNumberneuron() {
//        return numberNeuron;
//    }
//
//    public void setNumberneuron(int numberneuron) {
//        this.numberNeuron = numberneuron;
//    }

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

    public boolean getPredictStatus() {
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

    public void setTrainingCount(int trainingCount) {
        this.trainingCount = trainingCount;
    }

    public int getRound() {
        return round;
    }

    public void setRound(int round) {
        this.round = round;
    }

    public ArrayList<double[]> getInputsValuesReport() {
        return inputsValuesReport;
    }

    public void setInputsValuesReport(ArrayList<double[]> inputsValuesReport) {
        this.inputsValuesReport = inputsValuesReport;
    }

    public ArrayList<Double> getInitWeightsValuesReport() {
        return initWeightsValuesReport;
    }

    public void setInitWeightsValuesReport(ArrayList<Double> initWeightsValuesReport) {
        this.initWeightsValuesReport = initWeightsValuesReport;
    }

    public int getSamplePosition() {
        return samplePosition;
    }

    public void setSamplePosition(int samplePosition) {
        this.samplePosition = samplePosition;
    }

    public double getFunctionActivationResult() {
        return functionActivationResult;
    }

    public void setFunctionActivationResult(double functionActivationResult) {
        this.functionActivationResult = functionActivationResult;
    }

    public void setStructureInputReport(int numberNeuron, int numberLayer, String typeLayerName) {
        this.numberNeuronInput = numberNeuron;
        this.numberLayerInput = numberLayer;
        this.typeLayerNameInput = typeLayerName;
    }

    public void setStructureOutputReport(int numberNeuron, int numberLayer, String typeLayerName) {
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

    public ArrayList<Double> getInputReport() {
        return inputReport;
    }

    public void setInputReport(ArrayList<Double> inputReport) {
        this.inputReport = inputReport;
    }

    public int getnTraining() {
        return nTraining;
    }

    public void setnTraining(int nTraining) {
        this.nTraining = nTraining;
    }

}
package network;

import Help.Helper;
import core.Layer;
import core.NeuralNetwork;
import core.Neuron;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;


public class Mlp extends NeuralNetwork implements Serializable {
    private double learningRate;
    private double predict;
    private double bias;
    private Layer input;
    private Layer hidden;
    private Layer output;
    private ArrayList<Layer> hiddens;
    private double error;
    private double deltaB;
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
    //Teste
    private ArrayList<Double> predicts = new ArrayList<>();


    public Mlp(double learningRate, double predict, double bias) {
        this.learningRate = learningRate;
        this.predict = predict;
        this.bias = bias;
    }

    public Mlp(double learningRate, ArrayList<Double> predicts) {
        this.learningRate = learningRate;
        this.predicts = predicts;
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
        Random random = new Random();
        for (int i = 0; i < hidden.getNeuronsCount(); i++) {
            double n = Math.floor(random.nextDouble() * 100) / 100;
            bias_h.add(n);
//            System.out.println("bias hidden: " + bias_h.get(i));
        }

        for (int i = 0; i < output.getNeuronsCount(); i++) {
            double n = Math.floor(random.nextDouble() * 100) / 100;
            bias_o.add(n);
//            System.out.println("bias output: " + bias_o.get(i));
        }

    }

    @Override
    public void setInputValues(ArrayList inputValues) {
        samples = inputValues;
//        System.out.println("Valores da camada de entrada: ");
//        System.out.println("Qual é a lista da amostra: " + samplePositionList);
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).setInput(samples.get(samplePositionList)[i]);
//            System.out.print(input.getNeurons().get(i).getNetInput());
//            if (i % 28 == 0)
//                System.out.println();
        }
//        System.out.println("Posicao atual do inputvalues: " + samplePositionList);

    }

    public double getPredictValue(ArrayList predicts) {
//        System.out.println("Valor do samplesPosition " + predictPositionList);
//        System.out.println("Posicao atual do predicts: " + predictPositionList);
        return (double) predicts.get(predictPositionList);
    }

    public void setData(double[] data) {
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).setInput(data[i]);
//            System.out.println(input.getNeurons().get(i).getNetInput());
        }
    }

    @Override
    public void connectNeuronIncludingWeigth(double weigthValue) {
//        System.out.println("Conectando os neurônios");
        int i = 0;
        int k = 0;
//        System.out.println("Conectando da entrada até a oculta...");
//        System.out.println("Numero de neuronios na oculta: " + hidden.getNeuronsCount());
        while (i < input.getNeuronsCount()) {
            for (int j = 0; j < hidden.getNeuronsCount(); j++) {
                input.getNeurons().get(i).addInputConnection(hidden.getNeurons().get(j), weigthValue);
//                System.out.println("O neurônio: " + input.getNeurons().get(i).getNetInput() + " quantas conexões: " + input.getNeurons().get(i).getInputConnections().size() + " e está conectado com o neurônio posição: " + j);
            }
            i++;
        }
//        System.out.println("Conectando da oculta até a saída...");
//        System.out.println("Numero de neuronios na saida: " + output.getNeuronsCount());
        while (k < hidden.getNeuronsCount()) {
            for (int j = 0; j < output.getNeuronsCount(); j++) {
                hidden.getNeurons().get(k).addInputConnection(output.getNeurons().get(j), weigthValue);
//                System.out.println("O neurônio: " + hidden.getNeurons().get(k).getNetInput() + " quantas conexões: " + hidden.getNeurons().get(k).getInputConnections().size() + " e está conectado com o neurônio posição: " + j);
            }
            k++;
        }
    }

    @Override
    public void connectNeuronIncludingRandomWeigth() {
//        System.out.println("Conectando os neurônios");
        int i = 0;
        int k = 0;
        Random random = new Random();

//        System.out.println("Conectando da entrada até a oculta...");
//        System.out.println("Numero de neuronios na oculta: " + hidden.getNeuronsCount());
        while (i < input.getNeuronsCount()) {
            for (int j = 0; j < hidden.getNeuronsCount(); j++) {
//                System.out.println("numero aleatorio:" + random.nextDouble());
                double n = Math.floor(random.nextDouble() * 100) / 100;
                input.getNeurons().get(i).addInputConnection(hidden.getNeurons().get(j), n);
//                System.out.println("O neurônio: " + input.getNeurons().get(i).getNetInput() + " quantas conexões: " + input.getNeurons().get(i).getInputConnections().size() + " e está conectado com o neurônio posição: " + j);
            }
            i++;
        }
//        System.out.println("Conectando da oculta até a saída...");
//        System.out.println("Numero de neuronios na saida: " + output.getNeuronsCount());
        while (k < hidden.getNeuronsCount()) {
            for (int j = 0; j < output.getNeuronsCount(); j++) {
                double n = Math.floor(random.nextDouble() * 100) / 100;
                hidden.getNeurons().get(k).addInputConnection(output.getNeurons().get(j), n);
//                System.out.println("O neurônio: " + hidden.getNeurons().get(k).getNetInput() + " quantas conexões: " + hidden.getNeurons().get(k).getInputConnections().size() + " e está conectado com o neurônio posição: " + j);
            }
            k++;
        }
        randomizeBias();
    }

    @Override
    public void training() {
//        System.out.println("Start MLP!!!");
//        System.out.println("Start sum...");
        sum();
//        System.out.println("Start check outputs...");
        while (!checkOutputsMNIST()) {
//            System.out.println("A rede precisa de treinamento, resultado não corresponde com o esperado");
//            System.out.println("iniciando treinamento...");
            backpropagationTest();
            activateCounters();
            sum();
        }
//        System.out.println("Rede treinada!");
        checkNextSamples();
    }

    public void checkNextSamples() {
//        Helper.drawLine();
//        System.out.println("Checagem das próximas amostras");
        if (samplesCount != samples.size()) {
            trainWiththeNextSamples();
        } else {
            generateNewEpoch();
        }
    }

    @Override
    public void start() {
//        System.out.println("Start MLP!!!");
        sum();
//        for (int i = 0; i < output.getNeuronsCount(); i++) {
        if (checkOutputsMNIST()) {
//                System.out.println("O neuronio de posição " + i + " retornou: " + output.getNeurons().get(i).getOutput() + " e o valor esperado é: " + predict + " (SUCESSO)");
            System.out.println("Sucesso");
        } else {
//                System.out.println("O neuronio de posição " + i + " retornou: " + output.getNeurons().get(i).getOutput() + " e o valor esperado é: " + predict + " (FALHA)");
            System.out.println("Falha");
//            }
        }
    }

    public void sum() {
//        System.out.println("Realizando a somatória");
        int auxHidden = 0;
        int auxOutput = 0;
        double aux = 0;
//        ArrayList <Double> hiddenS = new ArrayList<>();
//        ArrayList <Double> outputS = new ArrayList<>();

        while (auxHidden < hidden.getNeuronsCount()) {
            for (int i = 0; i < input.getNeuronsCount(); i++) {
                aux += input.getNeurons().get(i).getNetInput() * input.getNeurons().get(i).getInputConnections().get(auxHidden).getWeight().getValue();
//                System.out.println("Valor da variavel aux da oculta: " + aux);
            }
            aux += bias_h.get(auxHidden);
//            hiddenS.add(auxHidden, aux + bias);
//            System.out.println("Valores da somatoria da camada oculta: " + hiddenS.get(auxHidden));
//            hidden.getNeurons().get(auxHidden).setInput(FunctionActivation.sigmoid(aux));
            selectFunctionActivationHidden(auxHidden, aux);
            auxHidden++;
            aux = 0;
        }

        while (auxOutput < output.getNeuronsCount()) {
            for (int i = 0; i < hidden.getNeuronsCount(); i++) {
                aux += hidden.getNeurons().get(i).getNetInput() * hidden.getNeurons().get(i).getInputConnections().get(auxOutput).getWeight().getValue();
//                System.out.println("Valor da variavel aux da saida: " + aux);
            }
            aux += bias_o.get(auxOutput);
//            outputS.add(auxOutput, aux + bias);
//            System.out.println("Valores da somatoria da camada saida: " + outputS.get(auxOutput));
//            output.getNeurons().get(auxOutput).setOutput(FunctionActivation.sigmoid(aux));
            selectFunctionActivationOutput(auxOutput, aux);
            auxOutput++;
            aux = 0;
        }
    }

    public void selectFunctionActivationHidden(int auxHidden, double aux) {
        if (getFunctionActivation().name().equals("DEGRAU")) {
            hidden.getNeurons().get(auxHidden).setInput(FunctionActivation.degrau(aux));
        } else if (getFunctionActivation().name().equals("SIGMOID")) {
            hidden.getNeurons().get(auxHidden).setInput(FunctionActivation.sigmoid(aux));
//            System.out.println("Valor da hidden dps da funcao de ativacao" + hidden.getNeurons().get(auxHidden).getNetInput());
        }
    }

    public void selectFunctionActivationOutput(int auxOutput, double aux) {
        if (getFunctionActivation().name().equals("DEGRAU")) {
            output.getNeurons().get(auxOutput).setOutput(FunctionActivation.degrau(aux));
        } else if (getFunctionActivation().name().equals("SIGMOID")) {
            //Math.round(media * 100.0)/100.0
//            output.getNeurons().get(auxOutput).setOutput(Math.floor(FunctionActivation.sigmoid(aux)*10000)/10000);
            output.getNeurons().get(auxOutput).setOutput(FunctionActivation.sigmoid(aux));
//            System.out.println("Valor do output dps da funcao de ativacao" + output.getNeurons().get(auxOutput).getNetInput());
        }
    }

    public boolean checkOutputs() {
        int s = 0;
        for (int i = 0; i < output.getNeuronsCount(); i++) {
            if (output.getNeurons().get(i).getOutput() == predict) {
                System.out.println("O neuronio de posição de saída: " + i + " retornou: " + output.getNeurons().get(i).getOutput() + " e o valor esperado é: " + predict + " (SUCESSO)");
                s++;
            } else {
                System.out.println("O neuronio de posição de saída: " + i + " retornou: " + output.getNeurons().get(i).getOutput() + " e o valor esperado é: " + predict + " (FALHA)");
            }
        }

        if (s > 0) {
            return true;
        } else {
            return false;
        }
    }

    public boolean checkOutputsMNIST() {
//        int s = 0;
        this.predictPositionList = samplePositionList;
        for (int i = 0; i < output.getNeuronsCount(); i++) {
            outputs.add(output.getNeurons().get(i).getOutput());
            System.out.println("Posição da lista: " + i + " Saída armazenada: " + outputs.get(i));
        }
//        System.out.println("max da lista: " + Collections.max(outputs) + " position: " + outputs.indexOf(Collections.max(outputs)));
        if (outputs.indexOf(Collections.max(outputs)) == getPredictValue(predicts)) {
//            System.out.println("Retornou: " + outputs.indexOf(Collections.max(outputs)) + " e o valor esperado é: " + predict + " (SUCESSO)");
            System.out.println("Retornou: " + outputs.indexOf(Collections.max(outputs)) + " e o valor esperado é: " + getPredictValue(predicts) + " (SUCESSO)");
//            s++;
            outputs.clear();
            return true;
        } else {
//            System.out.println("Retornou: " + outputs.indexOf(Collections.max(outputs)) + " e o valor esperado é: " + predict + " (FALHA)");
            System.out.println("Retornou: " + outputs.indexOf(Collections.max(outputs)) + " e o valor esperado é: " + getPredictValue(predicts) + " (FALHA)");
            outputs.clear();
            return false;
        }

    }

    public void trainWiththeNextSamples() {
//        Helper.drawLine();
//        System.out.println("Começando o treinamento!");
        samplePositionList++;
//        System.out.println("Posição da lista de amostras e de predicts: " + samplePositionList);
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
    }

//    public void backpropagation() {
//        //Apenas com uma saída
////        this.error = errorCalc(predict, output.getNeurons().get(0).getOutput());
//        //Com multiplas saídas
//        double sumError = 0;
//        for (int i = 0; i < output.getNeuronsCount(); i++) {
//            if (i == predict) {
////                System.out.println("Passar só uma vez");
//                sumError += (Math.pow(output.getNeurons().get(i).getOutput() - target, 2)) / 2;
//            }
//            sumError += (Math.pow(output.getNeurons().get(i).getOutput() - noTarget, 2)) / 2;
//        }
//        this.error = sumError;
//        System.out.println("Valor do erro..." + error);
//        System.out.println("Cálculo variação do bias...");
//        this.deltaB = deltaBiasCalc(error, learningRate);
//        System.out.println("Delta bias..." + deltaB);
//        System.out.println("Novo bias...");
//        this.bias = newBiasCalc(bias, deltaB);
//        System.out.println("Valor do novo bias..." + bias);
//        System.out.println("Cálculo variação do peso...");
//        this.deltaOW = new ArrayList<>();
//        int auxConnectionsHidden = 0;
//        while (auxConnectionsHidden < hidden.getNeuronsCount()) {
//            for (int i = 0; i < hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().size(); i++) {
//                deltaOW.add(i, deltaWeigthCalc(error, learningRate, hidden.getNeurons().get(auxConnectionsHidden).getNetInput()));
////                System.out.println("Valores inseridos dentro da lista deltaOW: " + deltaOW.get(i));
//            }
//            auxConnectionsHidden++;
//        }
//        this.deltaHW = new ArrayList<>();
//        int auxConnectionsInput = 0;
//        while (auxConnectionsInput < input.getNeuronsCount()) {
//            for (int i = 0; i < input.getNeurons().get(auxConnectionsInput).getInputConnections().size(); i++) {
//                deltaHW.add(i, deltaWeigthCalc(error, learningRate, input.getNeurons().get(auxConnectionsInput).getNetInput()));
////                System.out.println("Valores inseridos dentro da lista deltaHW: " + deltaHW.get(i));
//            }
//            auxConnectionsInput++;
//        }
//        auxConnectionsHidden = 0;
//        while (auxConnectionsHidden < hidden.getNeuronsCount()) {
////            System.out.println("Calculando novos pesos da saída até a oculta...");
//            for (int i = 0; i < hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().size(); i++) {
//                hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().setValue(newWeightCalc(hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().getValue(), deltaOW.get(auxConnectionsHidden)));
////                System.out.println("Valores dos novos pesos: " + hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().getValue());
//            }
////            System.out.println("auxConnectionsHidden: " + auxConnectionsHidden);
//            auxConnectionsHidden++;
//        }
//        auxConnectionsInput = 0;
//        while (auxConnectionsInput < input.getNeuronsCount()) {
////            System.out.println("Calculando novos pesos da oculta até a entrada...");
//            for (int i = 0; i < input.getNeurons().get(auxConnectionsInput).getInputConnections().size(); i++) {
//                input.getNeurons().get(auxConnectionsInput).getInputConnections().get(i).getWeight().setValue(newWeightCalc(input.getNeurons().get(auxConnectionsInput).getInputConnections().get(i).getWeight().getValue(), deltaHW.get(auxConnectionsInput)));
////                System.out.println("Valores dos novos pesos: " + input.getNeurons().get(auxConnectionsInput).getInputConnections().get(i).getWeight().getValue());
//            }
////            System.out.println("auxConnectionsInput: " + auxConnectionsInput);
//            auxConnectionsInput++;
//        }
//    }

    public void backpropagationTest() {
//        this.deltaB = deltaBiasCalc(error, learningRate);
//        System.out.println("Delta bias..." + deltaB);
//        this.bias = newBiasCalc(bias, deltaB);
//        System.out.println("Valor do novo bias..." + bias);

        //Oculta para a saída
        /*
        Formação do arraylist do calculo de cada erro (E)
         */
        ArrayList<Double> errorListO = new ArrayList<>();
        for (int i = 0; i < output.getNeuronsCount(); i++) {
//            if (i == predict) {
//                errorListO.add(target - output.getNeurons().get(i).getOutput());
//            }
            if (i == getPredictValue(predicts)) {
                errorListO.add(target - output.getNeurons().get(i).getOutput());
            }
            errorListO.add(noTarget - output.getNeurons().get(i).getOutput());
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
        ArrayList<Double> gradiente_O = new ArrayList<>();
        for (int i = 0; i < output.getNeuronsCount(); i++) {
            gradiente_O.add(i, errorListO.get(i) * errorListD_O.get(i));
        }

        /*
        formção do arraylist gradiente * learning rage (E o d(S) * lr)
         */
        ArrayList<Double> gradiente_lr_O = new ArrayList<>();
        for (int i = 0; i < output.getNeuronsCount(); i++) {
            gradiente_lr_O.add(i, gradiente_O.get(i) * this.learningRate);
        }

        /*
        ajuste do bias output
         */

        for (int i = 0; i < output.getNeuronsCount(); i++) {
            bias_o.set(i, bias_o.get(i) + gradiente_lr_O.get(i));
        }


        this.deltaOW = new ArrayList<>();
        int auxConnectionsHidden = 0;
        while (auxConnectionsHidden < hidden.getNeuronsCount()) {
            for (int i = 0; i < hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().size(); i++) {
                deltaOW.add(i, deltaWeigthCalc2(gradiente_lr_O.get(i), hidden.getNeurons().get(auxConnectionsHidden).getNetInput()));
//                System.out.println("Valor do i: " + i);
//                System.out.println("Valores inseridos dentro da lista deltaOW: " + deltaOW.get(i));
            }
            auxConnectionsHidden++;
        }
        auxConnectionsHidden = 0;
        while (auxConnectionsHidden < hidden.getNeuronsCount()) {
//            System.out.println("Calculando novos pesos da saída até a oculta...");
            for (int i = 0; i < hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().size(); i++) {
                hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().setValue(newWeightCalc(hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().getValue(), deltaOW.get(auxConnectionsHidden)));
//                System.out.println("Valores dos novos pesos: " + hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().getValue());
            }
//            System.out.println("auxConnectionsHidden: " + auxConnectionsHidden);
            auxConnectionsHidden++;
        }

//        //entrada para a oculta
        /*
        Formação do arraylist do calculo de cada erro
         */
        ArrayList<Double> errorListH = new ArrayList<>();
//        System.out.println("errorListH está vazio?: " + errorListH.isEmpty());
        auxConnectionsHidden = 0;
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
        ArrayList<Double> gradiente_H = new ArrayList<>();
        for (int i = 0; i < hidden.getNeuronsCount(); i++) {
            gradiente_H.add(i, errorListH.get(i) * errorListHiddenD.get(i));
        }
//        /*
//        formção do arraylist gradiente * learning rage
//         */
        ArrayList<Double> gradiente_lr_H = new ArrayList<>();
        for (int i = 0; i < hidden.getNeuronsCount(); i++) {
            gradiente_lr_H.add(i, gradiente_H.get(i) * this.learningRate);
        }

               /*
        ajuste do bias hidden
         */
        for (int i = 0; i < hidden.getNeuronsCount(); i++) {
            bias_h.set(i, bias_h.get(i) + gradiente_lr_H.get(i));
        }

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

        errorListO.clear();
        errorListD_O.clear();
        gradiente_O.clear();
        gradiente_lr_O.clear();
        deltaOW.clear();
        errorListH.clear();
        errorListHiddenD.clear();
        gradiente_H.clear();
        gradiente_lr_H.clear();
        deltaHW.clear();
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

    //Todo realizar as epochs
    //Todo solução para duas saídas
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

}

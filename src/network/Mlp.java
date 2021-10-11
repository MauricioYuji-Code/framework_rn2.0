package network;

import core.Layer;
import core.NeuralNetwork;
import core.Neuron;

import java.io.Serializable;
import java.lang.reflect.Type;
import java.util.ArrayList;


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


    public Mlp(double learningRate, double predict, double bias) {
        this.learningRate = learningRate;
        this.predict = predict;
        this.bias = bias;
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

    @Override
    public void setInputValues(ArrayList inputValues) {
        ArrayList<double[]> samples = inputValues;
        System.out.println("Valores da camada de entrada: ");
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).setInput(samples.get(0)[i]); //Exemplo apenas da primeira amostra
            System.out.println(input.getNeurons().get(i).getNetInput());
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
    public void training() {
        System.out.println("Start MLP!!!");
        System.out.println("Start sum...");
        sum();
        System.out.println("Start check outputs...");
        while (!checkOutputs()) {
            System.out.println("A rede precisa de treinamento, resultado não corresponde com o esperado");
            System.out.println("iniciando treinamento...");
            backpropagation();
            sum();
        }
        System.out.println("Rede treinada!");

    }

    @Override
    public void start() {
        System.out.println("Start MLP!!!");
        sum();
        for (int i = 0; i < output.getNeuronsCount(); i++) {
            if (checkOutputs()) {
                System.out.println("O neuronio de posição " + i + " retornou: " + output.getNeurons().get(i).getOutput() + " e o valor esperado é: " + predict + " (SUCESSO)");
            } else {
                System.out.println("O neuronio de posição " + i + " retornou: " + output.getNeurons().get(i).getOutput() + " e o valor esperado é: " + predict + " (FALHA)");
            }
        }
    }

    public void sum() {
        System.out.println("Realizando a somatória");
        int auxHidden = 0;
        int auxOutput = 0;
        double aux = 0;
//        ArrayList <Double> hiddenS = new ArrayList<>();
//        ArrayList <Double> outputS = new ArrayList<>();

        while (auxHidden < hidden.getNeuronsCount()) {
            for (int i = 0; i < input.getNeuronsCount(); i++) {
                aux += input.getNeurons().get(i).getNetInput() * input.getNeurons().get(i).getInputConnections().get(auxHidden).getWeight().getValue();
                System.out.println("Valor da variavel aux da oculta: " + aux);
            }
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
                System.out.println("Valore da variavel aux da saida: " + aux);
            }
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
        }
    }

    public void selectFunctionActivationOutput(int auxOutput, double aux) {
        if (getFunctionActivation().name().equals("DEGRAU")) {
            output.getNeurons().get(auxOutput).setOutput(FunctionActivation.degrau(aux));
        } else if (getFunctionActivation().name().equals("SIGMOID")) {
            output.getNeurons().get(auxOutput).setOutput(FunctionActivation.sigmoid(aux));
        }
    }

    public boolean checkOutputs() {
        int s = 0;
        for (int i = 0; i < output.getNeuronsCount(); i++) {
            if (output.getNeurons().get(i).getOutput() == predict) {
                System.out.println("O neuronio de posição de saída" + i + " retornou: " + output.getNeurons().get(i).getOutput() + " e o valor esperado é: " + predict + " (SUCESSO)");
                s++;
            } else {
                System.out.println("O neuronio de posição de saída" + i + " retornou: " + output.getNeurons().get(i).getOutput() + " e o valor esperado é: " + predict + " (FALHA)");
            }
        }

        if (s > 0) {
            return true;
        } else {
            return false;
        }
    }

    public void backpropagation() {
        //Apenas com uma saída
        this.error = errorCalc(predict, output.getNeurons().get(0).getOutput());
        System.out.println("Valor do erro..." + error);
        System.out.println("Cálculo variação do bias...");
        this.deltaB = deltaBiasCalc(error, learningRate);
        System.out.println("Delta bias..." + deltaB);
        System.out.println("Novo bias...");
        this.bias = newBiasCalc(bias, deltaB);
        System.out.println("Valor do novo bias..." + bias);
        System.out.println("Cálculo variação do peso...");
        this.deltaOW = new ArrayList<>();
        int auxConnectionsHidden = 0;
        while (auxConnectionsHidden < hidden.getNeuronsCount()) {
            for (int i = 0; i < hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().size(); i++) {
                deltaOW.add(i, deltaWeigthCalc(error, learningRate, hidden.getNeurons().get(auxConnectionsHidden).getNetInput()));
                System.out.println("Valores inseridos dentro da lista deltaOW: " + deltaOW.get(i));
            }
            auxConnectionsHidden++;
        }
        this.deltaHW = new ArrayList<>();
        int auxConnectionsInput = 0;
        while (auxConnectionsInput < input.getNeuronsCount()) {
            for (int i = 0; i < input.getNeurons().get(auxConnectionsInput).getInputConnections().size(); i++) {
                deltaHW.add(i, deltaWeigthCalc(error, learningRate, input.getNeurons().get(auxConnectionsInput).getNetInput()));
                System.out.println("Valores inseridos dentro da lista deltaHW: " + deltaHW.get(i));
            }
            auxConnectionsInput++;
        }
        auxConnectionsHidden = 0;
        while (auxConnectionsHidden < hidden.getNeuronsCount()) {
            System.out.println("Calculando novos pesos da saída até a oculta...");
            for (int i = 0; i < hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().size(); i++) {
                hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().setValue(newWeightCalc(hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().getValue(), deltaOW.get(auxConnectionsHidden)));
                System.out.println("Valores dos novos pesos: " + hidden.getNeurons().get(auxConnectionsHidden).getInputConnections().get(i).getWeight().getValue());
            }
            System.out.println("auxConnectionsHidden: " + auxConnectionsHidden);
            auxConnectionsHidden++;
        }

        auxConnectionsInput = 0;
        while (auxConnectionsInput < input.getNeuronsCount()) {
            System.out.println("Calculando novos pesos da oculta até a entrada...");
            for (int i = 0; i < input.getNeurons().get(auxConnectionsInput).getInputConnections().size(); i++) {
                input.getNeurons().get(auxConnectionsInput).getInputConnections().get(i).getWeight().setValue(newWeightCalc(input.getNeurons().get(auxConnectionsInput).getInputConnections().get(i).getWeight().getValue(), deltaHW.get(auxConnectionsInput)));
                System.out.println("Valores dos novos pesos: " + input.getNeurons().get(auxConnectionsInput).getInputConnections().get(i).getWeight().getValue());
            }
            System.out.println("auxConnectionsInput: " + auxConnectionsInput);
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

    public double deltaWeigthCalc(double e, double lr, double input) {
//        return Math.floor((e * lr * input) * 1000) / 1000;
        return e * lr * input;
    }

    public double newBiasCalc(double b, double deltaB) {
//        return Math.floor((deltaB + b) * 1000) / 1000;
        return deltaB + b;
    }

    public double newWeightCalc(double w, double deltaW) {
//        return Math.floor((deltaW + w) * 1000) / 1000;
        return deltaW + w;
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
}

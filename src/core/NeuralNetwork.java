package core;

import network.Perceptron;
import network.FunctionActivationData;
import utils.Report;

import java.io.*;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class NeuralNetwork implements Serializable {

    private List<Layer> layers;

    protected double[] outputBuffer;

    private List<Neuron> inputNeurons;

    private List<Neuron> outputNeurons;

    private String label = "";

    private Layer input;


    public NeuralNetwork() {
        this.layers = new ArrayList<>();
        this.inputNeurons = new ArrayList<>();
        this.outputNeurons = new ArrayList<>();
    }

    @Override
    public String toString() {
        if (label != null) {
            return label;
        }

        return super.toString();
    }


    //TODO

    public void setStructure(Type type, int nLayer, int nNeuron) {


    }

    public void setInputValues(ArrayList inputValues) {
        ArrayList<double[]> samples = inputValues;
        System.out.println("Valores da camada de entrada: ");
        for (int i = 0; i < input.getNeuronsCount(); i++) {
            input.getNeurons().get(i).setInput(samples.get(0)[i]); //Exemplo apenas da primeira amostra
            System.out.println(input.getNeurons().get(i).getNetInput());
        }
    }

    public void setInputWeights(double[][] inputWeights) {

    }

    public void setHiddenWeights(double[][] hiddenWeights) {

    }

    public void setActivationFunction(Type t) {

    }

    public void sigmoidFunction() {

    }

    public void start() {

    }

    public void training() {

    }

    public void connectNeuronIncludingWeigth(double weigthValue) {
    }

    public void connectNeuronIncludingRandomWeigth() {
    }

    public void save(String filePath) {

        ObjectOutputStream oos = null;

        try {
            File file = new File(filePath);

            oos = new ObjectOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
            oos.writeObject(this);
            oos.flush();
        } catch (FileNotFoundException e) {
            System.out.println("Arquivo não encontrado");
        } catch (IOException e) {
            System.out.println("Erro ao escrever a rede neural no arquivo");
        } finally {
            if (oos != null) {
                try {
                    oos.close();
                    System.out.println("Arquivo salvo");
                } catch (IOException e) {
                    System.out.println("Erro ao fechar o aquivo");
                }
            }
        }
    }

    public static NeuralNetwork load(String filePath) {

        ObjectInputStream ois = null;

        try {
            File file = new File(filePath);
            if (!file.exists()) {
                throw new FileNotFoundException("Arquivo não encontrado: " + filePath);
            }
            ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(filePath)));
            NeuralNetwork nn = (NeuralNetwork) ois.readObject();
            return nn;

        } catch (ClassNotFoundException e) {
            System.out.println("Classe não encontrada");
        } catch (IOException e) {
            System.out.println("Erro ao ler aquivo");
        } finally {
            if (ois != null) {
                try {
                    ois.close();
                    System.out.println("Arquivo carregado");
                } catch (IOException e) {
                    System.out.println("Erro ao fechar o aquivo");
                }
            }
        }
        return null;
    }

    public void setFunctionActivation(FunctionActivationData functionActivation) {

    }

    public Report getData() {
        return null;
    }

    public ArrayList<Perceptron> getReportFeedfoward() {
        return null;
    }

    public Map<Integer, Perceptron> getReportBackpropagation() {
        return null;
    }

    public void setData(double[] data) {

    }

    public void report(ArrayList<Perceptron> reports) {

    }

    public void setPredictValue(double predictValue) {
    }

    public void startFunctionActivation(int indicatorLayer, double resultSumNeuron, Layer destination, String label) {
    }
}

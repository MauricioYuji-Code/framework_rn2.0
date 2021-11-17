package main;

import core.*;
import mnist.MNISTInput;
import mnist.Mnist;
import mnist.MnistData;
import network.Mlp;
import network.Perceptron;
import network.FunctionActivationData;
import Help.Helper;
import utils.ImageU;
import utils.PixelCalc;
import utils.Report;
import utils.Type;

import java.io.IOException;
import java.util.ArrayList;

public class Main {

    public static void main(String[] args) throws IOException {
        //Perceptron
//        Helper.drawPerceptron();
        NeuralNetwork nn = new Perceptron(0.9, 0.8, 0);
        nn.setStructure(Type.INPUT, 1, 2);
        nn.setStructure(Type.OUTPUT, 1, 1);
        nn.setFunctionActivation(FunctionActivationData.SIGMOID);
        double sample1[] = {0, 1};
        double sample2[] = {1, 0};
        double sample3[] = {0, 0};
        ArrayList<double[]> list = new ArrayList<>();
        list.add(0, sample1);
        list.add(1, sample2);
        list.add(2, sample3);
        nn.setInputValues(list);
        nn.connectNeuronIncludingWeigth(0);
        nn.training();
//        nn.save("rede.rn");
//        //Perceptron start
//        NeuralNetwork perceptron = NeuralNetwork.load("rede.rn");
//        double data1[] = {1, 0};
//        double sample1[] = {0, 1, 0};
//        perceptron.setData(data1);
//        perceptron.start();

        //Data set MNIST
//        ArrayList<Number> mnistDataUnconverted = Mnist.generateDataMNIST();
//        MNISTInput mnistInput = new MNISTInput();
//        double[] mnistDataConverted = mnistInput.getInput(mnistDataUnconverted, 255, 0);

        MNISTInput mnistInput = new MNISTInput();
        MnistData[] mnistData = new Mnist().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        ArrayList <double[]> mnistSamples = new ArrayList<>();
        ArrayList <Double> mnistPredicts = new ArrayList<>();
        for (int i = 0 ; i < 2 ; i ++){
            double[] mnistDataConverted = mnistInput.getInputArray(mnistData[i], 255, 0);
//            double[] mnistDataConverted = mnistInput.getInputArray(mnistData[i]);
            mnistSamples.add(mnistDataConverted);
            mnistPredicts.add((double) mnistData[i].getLabel());
        }
//        System.out.println(mnistSamples.size());
//        System.out.println(mnistPredicts.size());
        double[] mnistDataConverted = mnistInput.getInputArray(mnistData[1], 255, 0);
//        System.out.println(mnistData[0].getLabel());

        //Teste com múltiplas entradas
//        NeuralNetwork nn3 = new Mlp(0.9, mnistPredicts);
//        nn3.setStructure(Type.INPUT, 1, 784);
//        nn3.setStructure(Type.HIDDEN, 1, 5);
//        nn3.setStructure(Type.OUTPUT, 1, 10);
//        nn3.connectNeuronIncludingRandomWeigth();
//        nn3.setFunctionActivation(FunctionActivationData.SIGMOID);
//        nn3.setInputValues(mnistSamples);
//        nn3.training();
//        nn3.save("rede-mlp.rn");
//        NeuralNetwork mlp = NeuralNetwork.load("rede-mlp-mnist.rn");
//        mlp.setData(mnistDataConverted);
//        mlp.start();


        //MLP treinamento
//        NeuralNetwork nn2 = new Mlp(0.9, mnistData[2].getLabel(), 0);
//        nn2.setStructure(Type.INPUT, 1, 784);
//        nn2.setStructure(Type.HIDDEN, 1, 10);
//        nn2.setStructure(Type.OUTPUT, 1, 10);
////        nn2.connectNeuronIncludingWeigth(0);
//        nn2.connectNeuronIncludingRandomWeigth();
//        nn2.setFunctionActivation(FunctionActivationData.SIGMOID);
//        ArrayList<double[]> samples = new ArrayList<>();
//        samples.add(0, mnistDataConverted);
//        nn2.setInputValues(samples);
//        nn2.training();
//        nn2.save("rede-mlp-mnist.rn");
        //MLP start
//        NeuralNetwork mlp = NeuralNetwork.load("rede-mlp-mnist.rn");
//        double data2[] = {0, 1};
//        mlp.setData(data2);
//        mlp.start();


        //Todo - Fazer relatório em HTML com os dados passados pelo parâmetro
//        Report.report(nn.getReports(), "myReport");

        //Todo - gráfico
//        PixelCalc pixelcalc = new PixelCalc();
//        ImageU imageu = new ImageU(pixelcalc, nn.getReportFeedfoward());
//        pixelcalc.setDate();
//        imageu.showImage();

    }
}

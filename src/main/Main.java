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
//        NeuralNetwork nn = new Perceptron(0.9, 0.8, 0);
//        nn.setStructure(Type.INPUT, 1, 2);
//        nn.setStructure(Type.OUTPUT, 1, 1);
//        nn.setFunctionActivation(FunctionActivationData.SIGMOID);
//        double sample1[] = {0, 1};
//        double sample2[] = {1, 0};
//        double sample3[] = {0, 0};
//        ArrayList<double[]> list = new ArrayList<>();
//        list.add(0, sample1);
//        list.add(1, sample2);
//        list.add(2, sample3);
//        nn.setInputValues(list);
//        nn.connectNeuronIncludingWeigth(0);
//        nn.training();
//        nn.save("rede.rn");
//        //Perceptron start
//        NeuralNetwork perceptron = NeuralNetwork.load("rede.rn");
//        double data1[] = {1, 0};
//        double sample1[] = {0, 1, 0};
//        perceptron.setData(data1);
//        perceptron.start();

        //Data set MNIST
        ArrayList<Number> mnistDataUnconverted = Mnist.generateDataMNIST();
        MNISTInput mnistInput = new MNISTInput();
        double[] mnistDataConverted = mnistInput.getInput(mnistDataUnconverted, 255, 0);


//        MnistData[] mnistData = new Mnist().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
//        double[] mnistDataConverted = mnistInput.getInput(mnistData[1], 255, 0);

        //MLP treinamento
        NeuralNetwork nn2 = new Mlp(0.9, 5, 0);
        nn2.setStructure(Type.INPUT, 1, 784);
        nn2.setStructure(Type.HIDDEN, 1, 10);
        nn2.setStructure(Type.OUTPUT, 1, 10);
//        nn2.connectNeuronIncludingWeigth(0);
        nn2.connectNeuronIncludingRandomWeigth();
        nn2.setFunctionActivation(FunctionActivationData.SIGMOID);
        ArrayList<double[]> samples = new ArrayList<>();
        samples.add(0, mnistDataConverted);
        samples.add(1, mnistDataConverted);
        nn2.setInputValues(samples);
        nn2.training();
//        nn2.save("rede-mlp.rn");
        //MLP start
//        NeuralNetwork mlp = NeuralNetwork.load("rede-mlp.rn");
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

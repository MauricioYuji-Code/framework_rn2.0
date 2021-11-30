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
//        double sample1[] = {0, 1};
//        double sample2[] = {1, 0};
//        double sample3[] = {0, 0};
//        double sample4[] = {1, 1};
//        ArrayList<double[]> list = new ArrayList<>();
//        list.add(0, sample1);
//        list.add(1, sample2);
//        list.add(2, sample3);
//        list.add(3, sample4);
//        ArrayList<Double> listPredicts = new ArrayList<>();
//        listPredicts.add(1.0);
//        listPredicts.add(1.0);
//        listPredicts.add(0.0);
//        listPredicts.add(1.0);

//        NeuralNetwork nn = new Perceptron(0.9, 0.8, 0);
//        NeuralNetwork nn = new Perceptron(0.9, listPredicts, 0);
//        nn.setStructure(Type.INPUT, 1, 2);
//        nn.setStructure(Type.OUTPUT, 1, 1);
//        nn.setFunctionActivation(FunctionActivationData.DEGRAU);
//        nn.setInputValues(list);
//        nn.connectNeuronIncludingWeigth(0);
//        nn.training();
//        nn.save("perceptron.rn");
//        //Perceptron start
//        NeuralNetwork perceptron = NeuralNetwork.load("perceptron.rn");
//        double data[] = {0, 0};
//        perceptron.setData(data);
//        perceptron.start();

        //Data set MNIST
        MNISTInput mnistInput = new MNISTInput();
        MnistData[] mnistData = new Mnist().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        ArrayList<double[]> mnistSamples = new ArrayList<>();
        ArrayList<Double> mnistPredicts = new ArrayList<>();
        for (int i = 0; i < 1; i++) {
//            double[] mnistDataConverted = mnistInput.getInputArray(mnistData[i], 255, 0);
//            double[] mnistDataConverted = mnistInput.getInputArray(mnistData[i]);
//            mnistSamples.add(mnistDataConverted);
            mnistSamples.add(mnistInput.getInputArray(mnistData[i]));
            mnistPredicts.add((double) mnistData[i].getLabel());
        }
//        double[] mnistDataConverted = mnistInput.getInputArray(mnistData[0], 255, 0);
        //Teste com múltiplas entradas
        NeuralNetwork nn3 = new Mlp(0.8, mnistPredicts);
        nn3.setStructure(Type.INPUT, 1, 784);
        nn3.setStructure(Type.HIDDEN, 1, 5);
        nn3.setStructure(Type.OUTPUT, 1, 10);
        nn3.connectNeuronIncludingRandomWeigth();
        nn3.setFunctionActivation(FunctionActivationData.SIGMOID);
        nn3.setInputValues(mnistSamples);
//        nn3.setNumberOfTrainings(10);
        nn3.training();
//        nn3.save("teste-rede-mlp-mnist2.rn");
//        NeuralNetwork mlp = NeuralNetwork.load("teste-rede-mlp-mnist2.rn");
//        mlp.setData(mnistSamples.get(0));
//        mlp.start();

        //            teste com degression
        double[] deg = {1.0, 1.0};
        double[] deg2 = {1.0, 0.0};
        double[] deg3 = {0.0, 0.0};
        double[] deg4 = {0.0, 1.0};
        ArrayList<double[]> degs = new ArrayList<>();
        degs.add(deg);
        degs.add(deg2);
//        degs.add(deg3);
//        degs.add(deg4);
        ArrayList<Double> predictsDegs = new ArrayList<>();
        predictsDegs.add(1.0);
        predictsDegs.add(1.0);
//        predictsDegs.add(0.0);
//        predictsDegs.add(1.0);

        NeuralNetwork nn4 = new Mlp(0.5, predictsDegs);
        nn4.setStructure(Type.INPUT, 1, 2);
        nn4.setStructure(Type.HIDDEN, 1, 2);
        nn4.setStructure(Type.OUTPUT, 1, 1);
        nn4.connectNeuronIncludingRandomWeigth();
        nn4.setFunctionActivation(FunctionActivationData.SIGMOID);
        nn4.setInputValues(degs);
//        nn4.setNumberOfTrainings(50);
        nn4.training();
//        nn4.save("rede-mlp-d-nok.rn");
//        NeuralNetwork mlp = NeuralNetwork.load("rede-mlp-d-nok.rn");
//        double[] testData = {0, 0};
//        mlp.setData(testData);
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

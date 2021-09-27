package main;

import core.*;
import network.Mlp;
import network.Perceptron;
import utils.FunctionActivationData;
import utils.Report;
import utils.Type;

import java.util.ArrayList;

public class Main {

    public static void main(String[] args) {
        //Perceptron
        NeuralNetwork nn = new Perceptron(0.01, 0, 0);
        nn.setStructure(Type.INPUT, 1, 2);
        nn.setStructure(Type.OUTPUT, 1, 1);
        nn.setFunctionActivation(FunctionActivationData.DEGRAU);
        double sample1[] = {0, 1};
        double sample2[] = {1, 0};
        ArrayList<double[]> list = new ArrayList<>();
        list.add(0, sample1);
        list.add(1, sample2);
        nn.setInputValues(list);
        nn.connectNeuronIncludingWeigth(0);
        nn.training();
//        nn.save("rede.rn");
//        //Perceptron start
//        NeuralNetwork perceptron = NeuralNetwork.load("rede.rn");
//        double data1[] = {0, 0};
//        perceptron.setData(data1);
//        perceptron.start();



        //MLP treinamento
//        NeuralNetwork nn2 = new Mlp(0.1, 0, 0);
//        nn2.setStructure(Type.INPUT, 1, 2);
//        nn2.setStructure(Type.HIDDEN, 1, 2);
//        nn2.setStructure(Type.OUTPUT, 1, 1);
//        double sample3[] = {0, 1};
//        double sample4[] = {1, 0};
//        ArrayList<double[]> list2 = new ArrayList<>();
//        list2.add(0, sample3);
//        list2.add(1, sample4);
//        nn2.setInputValues(list2);
//        nn2.connectNeuronIncludingWeigth(1);
//        nn2.training();
//        nn2.save("rede-mlp.rn");
        //MLP start
//        NeuralNetwork mlp = NeuralNetwork.load("rede-mlp.rn");
//        double input[] = {0, 1};
//        double input2[] = {0, 0};
//        ArrayList<double[]> inputs = new ArrayList<>();
//        inputs.add(0, input);
//        mlp.setInputValues(inputs);
//        mlp.start();

        //Teste do report
//        System.out.println("Teste dos dados: " + nn.getReports().get(0).getFunctionActivaion());
//        System.out.println("Teste dos dados: " + nn.getReports().get(1).getOutputValue());
//        System.out.println("Teste dos dados: " + nn.getReports().get(2).getOutputValue());


        //Todo - Fazer relatório em HTML com os dados passados pelo parâmetro
        Report.report(nn.getReports());
    }
}

package main;

import core.*;
import network.Perceptron;
import network.FunctionActivationData;
import Help.Helper;
import utils.ImageU;
import utils.PixelCalc;
import utils.Report;
import utils.Type;

import java.util.ArrayList;

public class Main {

    public static void main(String[] args) {
        //Perceptron
        Helper.drawPerceptron();
        NeuralNetwork nn = new Perceptron(0.01, 0, 0);
        nn.setStructure(Type.INPUT, 1, 2);
        nn.setStructure(Type.OUTPUT, 1, 1);
        nn.setFunctionActivation(FunctionActivationData.DEGRAU);
        double sample1[] = {0, 1};
        double sample2[] = {1, 0};
//        double sample3[] = {0, 0};
        ArrayList<double[]> list = new ArrayList<>();
        list.add(0, sample1);
        list.add(1, sample2);
//        list.add(2, sample3);
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


        //MLP treinamento
//        NeuralNetwork nn2 = new Mlp(0.9, 0.98, 0);
//        nn2.setStructure(Type.INPUT, 1, 3);
//        nn2.setStructure(Type.HIDDEN, 1, 2);
//        nn2.setStructure(Type.OUTPUT, 1, 1);
//        nn2.connectNeuronIncludingWeigth(1);
//        nn2.setFunctionActivation(FunctionActivationData.SIGMOID);
//        double sample4[] = {12, 3, 0.78};
////        double sample5[] = {0, 1, 0, 1, 6};
//        ArrayList<double[]> list2 = new ArrayList<>();
//        list2.add(0, sample4);
////        list2.add(1, sample5);
//        nn2.setInputValues(list2);
//        nn2.training();
//        nn2.save("rede-mlp.rn");
        //MLP start
//        NeuralNetwork mlp = NeuralNetwork.load("rede-mlp.rn");
//        double data2[] = {0, 1};
//        mlp.setData(data2);
//        mlp.start();

        //Teste do report

        System.out.println("Teste dos dados (Peso velho): " + nn.getReports().get(1).getInitWeightsValuesReport().get(1));
        System.out.println("Teste dos dados (Peso novo): " + nn.getReports().get(2).getNewWeightsValues().get(1));
        System.out.println("Teste dos dados (function activation result): " + nn.getReports().get(0).getFunctionActivationResult());
        System.out.println("Teste dos dados (structure input): " + nn.getReports().get(0).getStructureInputReport());
        System.out.println("Teste dos dados (structure output): " + nn.getReports().get(0).getStructureOutputReport());
        System.out.println("Teste dos dados (delta peso): " + nn.getReports().get(1).getDeltaWeightsValues().get(1));


//        System.out.println("Teste dos dados (Report): " + nn.getReports().get(1).getPredictStatus());
//        System.out.println("Teste dos dados (Report): " + nn.getReports().get(0).getInputsValuesReport().get(0)[0]);
//        System.out.println("Teste dos dados (Report): " + nn.getReports().get(0).getInputsValuesReport().get(0)[1]);

        //InputValues
//        int aux = 0;
//        String teste = "";
//        while (aux != nn.getReports().get(0).getInputsValuesReport().size()) {
//            for (int i = 0; i < nn.getReports().get(0).getInputsValuesReport().get(0).length; i++) {
//                System.out.println(nn.getReports().get(0).getInputsValuesReport().get(aux)[i]);
//                teste +=nn.getReports().get(0).getInputsValuesReport().get(aux)[i];
//            }
//            if (aux != nn.getReports().get(0).getInputsValuesReport().get(0).length){
//            }
//            aux++;
//        }
//        System.out.println(teste);

//        System.out.println("Teste dos dados: " + nn.getReports().get(0).getOutputValue());
//        System.out.println("Teste dos dados: " + nn.getReports().get(1).getOutputValue());
//        System.out.println("Teste dos dados: " + nn.getReports().get(2).getOutputValue());
//        System.out.println("Teste dos dados: " + nn.getReports().get(3).getOutputValue());

//        System.out.println("Epoca atual : " + nn.getReports().get(0).getEpoch());
//        System.out.println("Epoca atual : " + nn.getReports().get(1).getEpoch());
//        System.out.println("Epoca atual : " + nn.getReports().get(2).getEpoch());
//        System.out.println("Epoca atual : " + nn.getReports().get(3).getEpoch());
//        System.out.println("Epoca atual : " + nn.getReports().get(4).getEpoch());
//        System.out.println("Teste de rodadas: " + nn.getReports().get(0).getRound());
//        System.out.println("Teste de rodadas: " + nn.getReports().get(1).getRound());
//        System.out.println("Teste de rodadas: " + nn.getReports().get(2).getRound());
//        System.out.println("Teste de rodadas: " + nn.getReports().get(3).getRound());
//        System.out.println("Teste de rodadas: " + nn.getReports().get(4).getRound());


        //Todo - Fazer relatório em HTML com os dados passados pelo parâmetro
//        Report.report(nn.getReports(), "myReport");

        //Todo - gráfico
//
//        PixelCalc pixelcalc = new PixelCalc();
//        ImageU imageu = new ImageU(pixelcalc, nn.getReports());
//        pixelcalc.Dados();
//        imageu.criaImagem("Graph");

    }
}

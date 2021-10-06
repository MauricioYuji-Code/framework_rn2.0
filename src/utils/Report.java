package utils;

import Help.Helper;
import network.Perceptron;

import java.util.ArrayList;

public class Report {
    private ArrayList<Perceptron> reports;

    public Report(ArrayList<Perceptron> reports) {
        this.reports = reports;
    }

    public ArrayList<Perceptron> getReports() {
        return reports;
    }

    public void setReports(ArrayList<Perceptron> reports) {
        this.reports = reports;
    }

    //Todo Jean - Desenvoler o front nessa função
    public static void report(ArrayList<Perceptron> r) {

        for (int i = 0; i < r.size(); i++) {
            //Amostra atual
            System.out.println("Amostra " + (r.get(i).getSamplePosition()+1));
            //Epoca atual
            System.out.println("Epoca : " + r.get(i).getEpoch() + "\nIteração: " + r.get(i).getRound());
            //Saída atual
            System.out.println("Output: " + r.get(i).getOutputValue());
            //Numero de neuronios no perceptron
            System.out.println("Numero de Neuronios: " + (r.get(i).getNumberneuron()+1));
            //Status da predição
            System.out.println("Status da predição" + r.get(i).getPredictStatus());
            //Valor dos inputs
            System.out.println("Valores do input: " + r.get(0).getInputsValuesReport().get(0)[0]);
            System.out.println("Valores do input: " + r.get(1).getInputsValuesReport().get(0)[1]);
            System.out.println("Valores do input: " + r.get(0).getInputsValuesReport().get(1)[0]);
            System.out.println("Valores do input: " + r.get(1).getInputsValuesReport().get(1)[1]);
            //Valor dos pesos
            System.out.println("Valores do peso inicial: " + r.get(0).getInitWeightsValuesReport().get(0));
            System.out.println("Valores do peso inicial: " + r.get(0).getInitWeightsValuesReport().get(1));
            //Valor dos delta pesos
            System.out.println("Valores do delta pesos: " + r.get(1).getDeltaWeightsValues().get(0));
            System.out.println("Valores do delta pesos: " + r.get(1).getDeltaWeightsValues().get(1));
            //Valor dos novos pesos
            System.out.println("Valores do novo peso: " + r.get(1).getNewWeightsValues().get(0));
            System.out.println("Valores do novo peso: " + r.get(1).getNewWeightsValues().get(1));
            //Valor da somatória
            System.out.println("Valor da somatória: " + r.get(i).getSumValue());
            //Função de ativação
            System.out.println("Função de ativação: " + r.get(i).getFunctionActivaion().name());
            //Valor do erro
            System.out.println("Valor do erro: " + r.get(i).getErrorValue());
            //Valor do delta bias
            System.out.println("Valor do delta bias: " + r.get(i).getDeltaBias());
            //Valor do novo bias
            System.out.println("Valor do novo bias: " + r.get(i).getNewBias());
            for (int j = 0; j < r.get(i).getNumberneuron(); j++) {

            }
            Helper.drawLine();
        }
    }
}

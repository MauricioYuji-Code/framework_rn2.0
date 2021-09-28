package utils;

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

//            System.out.println("Treinamento: " + r.get(0).getRound());
//            System.out.println("Epoca: "+(r.get(0).getEpoch() + 1));


    }
}

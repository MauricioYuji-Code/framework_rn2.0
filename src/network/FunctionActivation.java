package network;

import java.io.Serializable;

public class FunctionActivation implements Serializable {

    public static double sigmoid(double n) {
//        System.out.println("Ativando a função Sigmoid");
//        return Math.floor((1 / (1 + Math.exp(-n))) * 100000) / 100000;
        return 1 / (1 + Math.exp(-n));
    }

    public static double sigmoidDer(double n) {
        double s = FunctionActivation.sigmoid(n);
        return s * (1 - s);
    }

    public static double degrau(double n) {
//        System.out.println("Ativando a função degrau!!");
        if (n >= 0) {
            return 1;
        } else {
            return 0;
        }
    }

    public static double relu(double n) {
        return Math.max(0, n);
    }

    public static double reluDer(double n) {
        if (n < 0)
            return 0;
        return 1;
    }

    public static double tanh(double n) {
        return Math.tanh(n);
    }

    public static double tanhDer(double n) {
        double tan = Math.tanh(n);
        return (1 / tan) - tan;
    }

    public static double leakyRelu(double n) {
        double result = 0;
        if (n > 0) result = n;
        else result = n * 0.01;
        return result;
    }

    public static double leakyReluDer(double x) {
        double result;
        if (x > 0) result = 1;
        else if (x < 0) result = 0.01;
        else result = 0;
        return result;
    }

}

package network;

import java.io.Serializable;

public class FunctionActivation implements Serializable {

    public static double sigmoid(double n) {
//        System.out.println("Ativando a função Sigmoid");
//        return Math.floor((1 / (1 + Math.exp(-n))) * 100000) / 100000;
        return 1 / (1 + Math.exp(-n));
    }

    public static double degrau(double n) {
//        System.out.println("Ativando a função degrau!!");
        if (n >= 0) {
            return 1;
        } else {
            return 0;
        }
    }

}

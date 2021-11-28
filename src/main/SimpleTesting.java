package main;


import java.util.Arrays;
import java.util.Random;

public class SimpleTesting {
    public static void main(String[] args) {
        Random random = new Random();
        double n = Math.floor(random.nextDouble() * 100) / 100;
        System.out.println(n);
    }
}

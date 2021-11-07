package mnist;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.*;
import java.util.ArrayList;

public class Mnist {

//    public Mnist() {
//
//    }

    public  MnistData[] readData(String filePathTrain, String filePathLabel) throws IOException {
        //TREINAMENTO DEFINIR ARQUIVO DE IMAGEM (train-images-idx3-ubyte):
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(filePathTrain)));
//        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream("data/test.wav")));
        //Byte mais significativo
        int numeroMagico = dataInputStream.readInt();
        int numeroDeImagens = dataInputStream.readInt();
        int linhas = dataInputStream.readInt();
        int colunas = dataInputStream.readInt();

        System.out.println("Número mágico: " + numeroMagico);
        System.out.println("Número de imagens: " + numeroDeImagens);
        System.out.println("Linhas: " + linhas);
        System.out.println("Colunas: " + colunas);

        //TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(filePathLabel)));
        int digito = labelInputStream.readInt();
        int numeroDeDigitos = labelInputStream.readInt();

        System.out.println(digito);
        System.out.println(numeroDeDigitos);
        MnistData[] data = new MnistData[numeroDeImagens];
        //Formato vetor
        for (int i = 0; i < numeroDeImagens; i++) {
            MnistData mnistData = new MnistData(linhas * colunas);
            mnistData.setLabel(labelInputStream.readUnsignedByte());
//            System.out.println(mnistData.getLabel());
            for (int j = 0; j < (linhas * colunas); j++) {
//                System.out.println(j);
                mnistData.setValueOfArray(j, dataInputStream.readUnsignedByte());
            }
            data[i] = mnistData;
        }

        dataInputStream.close();
        labelInputStream.close();
        return data;
    }

    public static ArrayList generateDataMNIST() throws IOException {
        //TREINAMENTO DEFINIR ARQUIVO DE IMAGEM (train-images-idx3-ubyte):
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream("data/train-images.idx3-ubyte")));
//        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream("data/test.wav")));
        //Byte mais significativo
        int numeroMagico = dataInputStream.readInt();
        int numeroDeImagens = dataInputStream.readInt();
        int linhas = dataInputStream.readInt();
        int colunas = dataInputStream.readInt();

        System.out.println("Número mágico: " + numeroMagico);
        System.out.println("Número de imagens: " + numeroDeImagens);
        System.out.println("Linhas: " + linhas);
        System.out.println("Colunas: " + colunas);

        //TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
        DataInputStream labelInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream("data/train-labels.idx1-ubyte")));
        int digito = labelInputStream.readInt();
        int numeroDeDigitos = labelInputStream.readInt();

        System.out.println(digito);
        System.out.println(numeroDeDigitos);

        //Formato bit map
//        int[][] bitMap = new int[linhas][colunas];
//        for (int r = 0; r < linhas; r++) {
//            for (int c = 0; c < colunas; c++) {
//                bitMap[r][c] = dataInputStream.readByte();
//                System.out.print(bitMap[r][c] + " ");
//            }
//            System.out.println(" ");
//        }
        //Formato vetor
        ArrayList<Number> data = new ArrayList<>();
            for (int j = 0; j < (linhas * colunas); j++) {
                data.add(dataInputStream.readUnsignedByte());
        }
//        MNISTInput mnistInput = new MNISTInput();
//        mnistInput.getInput(data, 255, 0);
        dataInputStream.close();
        labelInputStream.close();
        return data;
    }
}
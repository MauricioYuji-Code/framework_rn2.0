package mnist;

import java.io.IOException;

public class MnistTest {

    public static void main(String[] args) throws IOException {

        MnistData[] mnistData = new Mnist().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
//        printMnistMatrix(mnistData[mnistData.length-1]);
        System.out.println("Data size: " + mnistData.length);
        printMnistMatrix(mnistData[1]);

    }

    private static void printMnistMatrix(MnistData matrix) {
        System.out.println("Digito: " + matrix.getLabel());
        for (int r = 0; r < matrix.getSizeOfArray(); r++) {
            System.out.print(matrix.getValueOfArray(r) + " ");
            if (r % 28 == 0)
                System.out.println();
        }
    }
}

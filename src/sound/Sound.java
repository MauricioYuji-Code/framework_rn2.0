package sound;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;

public class Sound {

    private static String toBin(int i) {
        String temp = "00000000" + Integer.toBinaryString(i);
        System.out.println(temp);
        return temp.substring((temp.length() - 8)).toUpperCase();
    }

    private static int toDec(String i) {
        return Integer.parseInt(i, 2);
    }

    private static byte[] intToBytes(final int data) {
        return new byte[] {
                (byte)((data >> 24) & 0xff),
                (byte)((data >> 16) & 0xff),
                (byte)((data >> 8) & 0xff),
                (byte)((data >> 0) & 0xff),
        };
    }

    public static void main(String[] args) throws IOException {
//        File file = new File("data/test.wav");
        File file = new File("data/Test_image.jpg");
        DataInputStream dataInputStream = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
        byte[] inputData = new byte[(int) file.length()];
        int[] outputIntData = new int[(int) file.length()];
        byte[] outputByteData = new byte[(int) file.length()];
        System.out.println("Tamanho do arquivo: " + file.length());
        for (int c = 0; c < inputData.length; c++) {
//            outputIntData[c] = dataInputStream.read();
//            System.out.print("|" + outputIntData[c]);
            outputByteData[c] = (byte) dataInputStream.read();
            System.out.print("|" + outputByteData[c]);
        }

//        System.out.println("Tamanho do arquivo: " + dataInputStream.readAllBytes());

//        FileInputStream in = new FileInputStream(file);
//        in.read(inputData);
//        in.close();

//         Fill the buffer with binary numbers
//        for (int x : inputData) {
//            String binaryNumber = toBin(x);
//            System.out.println("x=" + x + ", binaryNumber=" + binaryNumber);
//            int decimalNumber = toDec(binaryNumber);
//            System.out.println("x=" + x + ", decimalNumber=" + decimalNumber);
////            sb.append(binaryNumber);
//        }
        FileOutputStream out = new FileOutputStream("data/output.jpg");
        out.write(outputByteData);
        out.close();

        // Split the buffer into chunks of 8 characters
//        for (int i = 0; i < sb.length(); i += 8) {
//            String part = sb.substring(i, i + 8);
//
//            // Convert back to integer
//            int x = Integer.parseInt(part, 2);
//
//            System.out.println("part=" + part + ", x=" + x);
//        }


    }
}
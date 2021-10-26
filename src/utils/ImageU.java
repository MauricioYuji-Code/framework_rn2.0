package utils;

import network.Perceptron;
import utils.PixelCalc;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.awt.RenderingHints;


public class ImageU {

    double x0;
    double x1;
    double y0;
    double y1;

    double x;
    double oix;
    double ofx;
    double dix;
    double dfx;
    double pxr;

    double y;
    double oiy;
    double ofy;
    double diy;
    double dfy;
    double pyr;

    int xAxis;
    int yAxis;

    String xLabel;
    String yLabel;

    double percentage;

    BufferedImage report;
    Graphics2D render;
    PixelCalc pixelcalc;

    List<Double> auxList;
    List<Point> pointList;


    ArrayList<Perceptron> datas;

    public ImageU(PixelCalc pixelcalc, ArrayList<Perceptron> datas) {
        this.pixelcalc = pixelcalc;
        this.datas = datas;
    }


    public double getMaxValue() {
        double max = Double.MIN_VALUE;
        for (Double aDouble : auxList) {
            if (aDouble > max) {
                max = aDouble;
            }
        }
        return max;
    }

    public double getMinValue() {
        double min = Double.MAX_VALUE;
        for (Double aDouble : auxList) {
            if (aDouble < min) {
                min = aDouble;
            }
        }
        return min;
    }

    public void background() {
        if (pixelcalc.getPixels_width() == null && pixelcalc.getPixels_height() == null) {
            pixelcalc = new PixelCalc();
            pixelcalc.defaultImage();
        }
        report = new BufferedImage(pixelcalc.getPixels_width(), pixelcalc.getPixels_height(), BufferedImage.TYPE_INT_RGB);
        render = (Graphics2D) report.getGraphics();
        render.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        render.setColor(Color.lightGray);
        render.fillRect(0, 0, report.getWidth(), report.getHeight());
    }

    public double label() {
        return report.getWidth() * this.percentage;
    }

    public void whiteBackground() {

        this.percentage = 0.05;
        x0 = label();
        x1 = report.getWidth() - label() - 1;
        y0 = label();
        y1 = report.getHeight() - 1;

        render.setColor(Color.WHITE);
        render.fillRect((int) x0, (int) y0 - 15, (int) x1 - (int) x0 + 15, (int) y1 - (int) y0);
        setTitle();
    }

    public void setTitle() {
        render.setColor(Color.BLACK);
        render.drawString("Graphic Neural Network: TEST", (int) x0, (int) y0 - 20);
    }


    public List<Double> valuesSaida() {
        auxList = new ArrayList<>();
        for (int i = 0; i < datas.size(); i++) {
            auxList.add(datas.get(i).getOutputValue());
        }
        return auxList;
    }

    public double xMapping_array() {
        pxr = ((x - oix) * (dfx - dix) / (ofx - oix)) + dix;
        return pxr;
    }

    public double xMapping_plot() {
        pxr = (x * (ofx - oix) / (dfx - dix)) + dix;
        return pxr;
    }

    public void setGrid_subtitle() {
        render.setColor(Color.lightGray);
        render.drawLine((int) pxr, (int) (y1 - label()), (int) pxr, (int) y0);

        if ((double) auxList.size() % ofx == 0) {
            xLabel = Math.round(x) + "";
        } else {
            xLabel = Math.round(x * 100.0) / 100.0 + "";
        }

        FontMetrics metrics = render.getFontMetrics();
        int labelWidth_x = metrics.stringWidth(xLabel);

        render.setColor(Color.BLACK);
        render.drawString(xLabel, (int) pxr - labelWidth_x / 2, (int) y1 - (int) label() + metrics.getHeight() + 5);
    }

    public void yMapping() {
        pyr = ((y - ofy) * (diy - dfy) / (oiy - ofy)) + dfy;
    }

    public void setAxis() {
        percentage = 0.1;

        render.setColor(Color.BLACK);
        render.drawLine((int) x0 + 75, (int) y1 - (int) label(), (int) x1, (int) y1 - (int) label()); // axis x
        render.drawLine((int) x0 + (int) label(), (int) y0, (int) x0 + (int) label(), (int) y1 - 75); // axis Y
    }

    public void setGrid() {

        Stroke oldStroke = render.getStroke();
        float[] dash = {2f, 0f, 2f};
        BasicStroke bs = new BasicStroke(1, BasicStroke.CAP_BUTT, BasicStroke.JOIN_ROUND, 1.0f, dash, 2f);
        render.setStroke(bs);

        percentage = 0.1;

        oix = 0;
        ofx = auxList.size();
        dix = (int) (x0 + label());
        dfx = x1;

        oiy = getMaxValue();
        ofy = getMinValue();
        diy = y0;
        dfy = (int) (y1 - label());

        x = 0;
        if (auxList.size() > 10) {
            for (int i = 0; i <= auxList.size(); i++) {

                xMapping_array();
                x++;

                int varAux = auxList.size() / 10;
                if (auxList.size() % 2 == 0) {
                    if (x % varAux == 0)
                        setGrid_subtitle();
                } else
                    setGrid_subtitle();
            }
        } else
            for (int i = 0; i <= auxList.size(); i++) {
                x = i;
                xMapping_array();
                setGrid_subtitle();
            }


        //Y
        y = getMinValue();
        for (int j = 0; j < 11; j++) {

            yMapping();

            render.setColor(Color.lightGray);
            render.drawLine((int) (x0 + label()), (int) pyr, (int) x1, (int) pyr);


            yLabel = Math.round(y * 100.00) / 100.00 + "";


            FontMetrics metrics = render.getFontMetrics();
            int labelWidth_y = metrics.stringWidth(yLabel);

            render.setColor(Color.BLACK);
            render.drawString(yLabel, (int) x0 + (int) label() - labelWidth_y - 10, (int) pyr + metrics.getHeight() / 2);

            y += ((getMaxValue() - getMinValue()) / 10);

        }
        render.setStroke(oldStroke);
    }

    public void setLinePoints() {
        pointList = new ArrayList<>();
        percentage = 0.1;

        oix = 0;
        ofx = auxList.size();
        dix = (int) (x0 + label());
        dfx = x1;

        oiy = getMaxValue();
        ofy = getMinValue();
        diy = y0;
        dfy = (int) (y1 - label());


//        if (auxList.size() < (dfx - dix)) {
//            for (int i = 0; i < auxList.size(); i++) {
//                x = i;
//                xMapping_array();
//
//            }
//        } else {
//            for (int i = 0; i < (dfx - dix); i++) {
//                x = i;
//                xMapping_plot();
//            }
//            setGrid_subtitle();
//        }
//            xMapping_plot();


        for (int i = 0; i < auxList.size(); i++) {
            x = i;
            xMapping_array();
//            if (i < auxList.size())
            y = auxList.get(i);
            yMapping();
            pointList.add(new Point((int) pxr, (int) pyr));

        }

        for (int j = 0; j < pointList.size() - 1; j++) {
            render.setColor(Color.blue);
            render.drawLine(pointList.get(j).x, pointList.get(j).y, pointList.get(j + 1).x, pointList.get(j + 1).y);
        }
    }

    public void setPoints() {

        percentage = 0.1;

        oix = 0;
        ofx = auxList.size();
        dix = (int) (x0 + label());
        dfx = x1;

        oiy = getMaxValue();
        ofy = getMinValue();
        diy = y0;
        dfy = (int) (y1 - label());

        for (int i = 0; i < auxList.size(); i++) {
            x = i;
            xMapping_array();

//            if (i < auxList.size())
            y = auxList.get(i);
            yMapping();

            render.setColor(Color.RED);
            render.fillOval((int) pxr - 4, (int) pyr - 4, 8, 8);

        }
    }

    public void saveImage() {
        try {
            ImageIO.write(report, "PNG", new File("test.png"));
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
    }

    public void showImage() {
        valuesSaida();
        background();
        whiteBackground();
        setGrid();
        setAxis();
        setLinePoints();
        setPoints();
        saveImage();
    }

}


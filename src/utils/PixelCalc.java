package utils;

public class PixelCalc {

    Integer Ratio_width = 4;
    Integer Ratio_height = 3;
    Integer Pixels_width = null;
    Integer Pixels_height = null;
    String aux = "";

    public void defaultImage (){
        this.Pixels_width = 800;
        this.Pixels_height = 600;
    }

    public void setPixelsWidth (int Pixels_width){
        this.Pixels_width = Pixels_width;
    }

    public void setPixelsHeight (int Pixels_height){
        this.Pixels_height = Pixels_height;
    }

    public void setRatio (int Ratio_width,int Ratio_height){
        this.Ratio_width = Ratio_width;
        this.Ratio_height = Ratio_height;
    }

    public Integer getRatio_width() {
        return Ratio_width;
    }

    public Integer getRatio_height() {
        return Ratio_height;
    }

    public Integer getPixels_width() {
        return Pixels_width;
    }

    public Integer getPixels_height() {
        return Pixels_height;
    }

    public int proporcao() {
        if (Pixels_height != 0) {
            Pixels_width = (Pixels_height / Ratio_height) * Ratio_width;
            return Pixels_width;
        }
        else if (Pixels_width == 0 && Pixels_height == 0){
            Pixels_width = 800;
            Pixels_height = (Pixels_width / Ratio_width) * Ratio_height;
            return Pixels_height;
        }
        else{
            Pixels_height = (Pixels_width / Ratio_width) * Ratio_height;
            return Pixels_height;
        }
    }

    public String Dados (){
        if(Pixels_width == null && Pixels_height == null)
            defaultImage();
        System.out.println("Ratio " + getRatio_width() + ":" + getRatio_height());
        System.out.println(aux = "Largura: " + Pixels_width + " Altura: " + Pixels_height);
        return aux;
    }

}
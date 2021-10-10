package utils;

import Help.Helper;
import network.Perceptron;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
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

    public static String generateHTML(String report_name, String content) {
        String html_head = "<!DOCTYPE html>\n" +
                "<html lang='pt-br'>\n" +
                "<head>\n" +
                "  <meta charset='UTF-8'>\n" +
                "  <meta http-equiv='X-UA-Compatible' content='IE=edge'>\n" +
                "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n" +
                "  <title> " + report_name + "</title>\n" +
                "</head>\n";

        String html_header = "  <header>\n" +
                "    <div class='header-content-container'>\n" +
                "      <h1>Trainning Report</h1>\n" +
                "      <span>PROJECT: " + report_name + "</span>\n" +
                "    </div>\n" +
                "  </header>\n";

        String html_body_openning_tag = "<body>\n";

        String html_container_openning_tag = "<div class='container'>\n";
        String html_container_closing_tag = "</div>\n";



        String header_content = "  <div class='container'>\n" +
                "    <div class='nn-data'>\n" +
                "      <div class='nn-initial-info'>\n" +
                "        <strong>Training Data:</strong>\n" +
                "        <ul>\n" +
                "          <!-- Criar estrutura diferente para os pesos e as amostras, pois são arrays e a forma e exibir os dados é diferente -->\n" +
                "          <li><span>Samples: </span></li>\n" +
                "          <li><span>Weights: </span></li>\n" +
                "          <li><span>Learning Rate: </span></li>\n" +
                "          <li><span>Predict: </span></li>\n" +
                "          <li><span>Bias: </span></li>\n" +
                "          <li><span>Structure: </span></li>\n" +
                "          <li><span>Network Type: </span></li>\n" +
                "          <li><span>Activation Function: </span></li>\n" +
                "        </ul>\n" +
                "      </div>\n" +
                "\n" +
                "      <div class='training-execution-data'>\n" +
                "        <strong>Training cycle statics:</strong>\n" +
                "        <ul>\n" +
                "          <li><span>Training numbers: </span></li>\n" +
                "          <li><span>Batch size: </span></li>\n" +
                "          <li><span>Iterations: </span></li>\n" +
                "        </ul>\n" +
                "      </div>\n" +
                "    </div>\n";


        String html_body_closing_tag = "</body>\n";

        String css = "<style>\n" +
                "    * {\n" +
                "      margin: 0;\n" +
                "      padding: 0;\n" +
                "      box-sizing: border-box;\n" +
                "      outline: none !important;\n" +
                "    }\n" +
                "\n" +
                "    body {\n" +
                "      display: flex;\n" +
                "      flex-direction: column;\n" +
                "      font-family: sans-serif;\n" +
                "    }\n" +
                "\n" +
                "    html,\n" +
                "    body {\n" +
                "      min-height: 100vh;\n" +
                "      background-color: #f2f2f2;\n" +
                "    }\n" +
                "\n" +
                "    header {\n" +
                "      display: flex;\n" +
                "      width: 100%;\n" +
                "      justify-content: center;\n" +
                "      align-items: center;\n" +
                "      margin-bottom: 30px;\n" +
                "    }\n" +
                "\n" +
                "    .container {\n" +
                "      display: flex;\n" +
                "      flex-direction: column;\n" +
                "      width: 100%;\n" +
                "      max-width: 1160px;\n" +
                "      margin: 0 auto;\n" +
                "      padding: 0 20px;\n" +
                "    }\n" +
                "\n" +
                "    .header-content-container {\n" +
                "      display: flex;\n" +
                "      width: 100%;\n" +
                "      justify-content: center;\n" +
                "      align-items: center;\n" +
                "      max-width: 1160px;\n" +
                "      padding: 20px;\n" +
                "      flex-direction: column;\n" +
                "    }\n" +
                "\n" +
                "    .header-content-container h1 {\n" +
                "      margin-bottom: 10px;\n" +
                "    }\n" +
                "\n" +
                "    .nn-data {\n" +
                "      display: flex;\n" +
                "      width: 100%;\n" +
                "    }\n" +
                "\n" +
                "    .nn-data>div {\n" +
                "      width: 50%;\n" +
                "    }\n" +
                "\n" +
                "    .nn-data>div {\n" +
                "      padding-right: 20px;\n" +
                "    }\n" +
                "\n" +
                "    @media(max-width: 768px) {\n" +
                "      .nn-data {\n" +
                "        flex-direction: column;\n" +
                "      }\n" +
                "\n" +
                "      .nn-data>div {\n" +
                "        width: 100%;\n" +
                "        padding: 0;\n" +
                "      }\n" +
                "    }\n" +
                "\n" +
                "    .nn-initial-info {\n" +
                "      display: flex;\n" +
                "      width: 100%;\n" +
                "      flex-direction: column;\n" +
                "      margin-bottom: 30px;\n" +
                "    }\n" +
                "\n" +
                "    .nn-initial-info strong {\n" +
                "      display: flex;\n" +
                "      font-size: 22px;\n" +
                "      margin-bottom: 15px;\n" +
                "    }\n" +
                "\n" +
                "    .nn-initial-info ul,\n" +
                "    .training-execution-data ul {\n" +
                "      display: flex;\n" +
                "      flex-direction: column;\n" +
                "      width: 100%;\n" +
                "      list-style: none;\n" +
                "    }\n" +
                "\n" +
                "    .nn-initial-info ul li,\n" +
                "    .training-execution-data ul li {\n" +
                "      margin-bottom: 5px;\n" +
                "    }\n" +
                "\n" +
                "    .training-execution-data strong {\n" +
                "      display: flex;\n" +
                "      font-size: 22px;\n" +
                "      margin-bottom: 15px;\n" +
                "    }\n" +
                "\n" +
                "    .nn-iteration-container {\n" +
                "      display: flex;\n" +
                "      flex-direction: column;\n" +
                "      width: 100%;\n" +
                "      border: 2px solid black;\n" +
                "      margin-bottom: 30px;\n" +
                "    }\n" +
                "\n" +
                "    .nn-iteration-content-container {\n" +
                "      display: flex;\n" +
                "      flex-direction: column;\n" +
                "    }\n" +
                "\n" +
                "    .nn-iteration-content-container span {\n" +
                "      margin-bottom: 10px;\n" +
                "      margin-right: 10px;\n" +
                "    }\n" +
                ".nn-iteration-container > strong:first-child {\n" +
                "    font-weight: bold;\n" +
                "    padding: 15px 20px;\n" +
                "    border-bottom: 2px solid black;\n" +
                "}" +
                ".nn-iteration-container > strong:nth-child(2) {\n" +
                "    margin-bottom: 15px;\n" +
                "    padding: 0 20px;\n" +
                "}" +
                "\n" +
                ".feedworward-container, .backpropagation-container {\n" +
                "    display: flex;\n" +
                "    flex-direction: column;\n" +
                "    margin-bottom: 10px;\n" +
                "    padding: 20px;\n" +
                "}" +
                "    footer {\n" +
                "      margin-top: auto;\n" +
                "    }\n" +
                "  </style>";

        String html_close_tag = "</html>";


        return html_head + html_header + html_body_openning_tag  + html_container_openning_tag + header_content + content + html_container_closing_tag + css + html_body_closing_tag + html_close_tag;
    }


    public static void reportPrint(String filePath, String content) {

        FileWriter fw = null;
        try {
            fw = new FileWriter(new File(filePath + ".html"));
            fw.write(content);
            fw.flush();
            fw.close();
        } catch (Exception e) {
            System.out.println("Erro bagulho doido");
        } finally {
            if (fw != null) {
                try {
                    fw.close();
                    System.out.println("Arquivo salvo");
                } catch (IOException e) {
                    System.out.println("Erro ao fechar o aquivo");
                }
            }
        }
    }


    public static String generateReportContentHTMl(ArrayList<Perceptron> r) {
        String nn_content = "";
        for (int i = 0; i < r.size(); i++) {
            nn_content += "<div class='nn-iteration-container'>" +
                    "<strong>Época " + r.get(i).getEpoch() + "- Amostra " + (r.get(i).getSamplePosition()+1) + "</strong>" +
                    "<div class='nn-iteration-content-container'>" +
                    "<div class='feedworward-container'>" +
                    "<strong style='margin:15px 0'>Feedforward</strong>" +
                    "<span>Input Layer: valores da camada de entrada aqui </span>" +
                    "<span>Pesos (Input layer): valores dos pesos aqui</span>" +
                    "<span>Valor da somatória: " +  r.get(i).getSumValue() + "</span>" +
                    "<span>Resultado da função de ativação: valor da função de ativação vai aqui</span>" +
                    "<span>Output: " + r.get(i).getOutputValue() + "</span>" +
                    "<span>Predict: " +  r.get(i).getPredictStatus() + "</span>" +
                    "</div>" +
                    "<div class='backpropagation-container'>" +
                    "<strong style='margin:15px 0'>Backpropagation</strong>" +
                    //"<span>Número de neuronios no perceptron: " +  (r.get(i).getNumberneuron()+1) + "</span>" +
                    "<span>Valor do erro: " +  r.get(i).getErrorValue() + "</span>" +
                    "<span>Delta peso: valor do delta peso aqui </span>" +
                    "<span>Novo peso: valor do novo peso aqui</span>" +
                    "<span>Delta bias: " +  r.get(i).getDeltaBias() + "</span>" +
                    "<span>Novo bias: " +  r.get(i).getNewBias() + "</span>" +
                    "</div></div></div>";

//            System.out.println(nn_content);


            //Amostra atual
            //            System.out.println("Amostra " + (r.get(i).getSamplePosition()+1));
            //            //Epoca atual
            //            System.out.println("Epoca : " + r.get(i).getEpoch() + "\nIteração: " + r.get(i).getRound());
            //            //Saída atual
            //            System.out.println("Output: " + r.get(i).getOutputValue());
            //            //Numero de neuronios no perceptron
            //            System.out.println("Numero de Neuronios: " + (r.get(i).getNumberneuron()+1));
            //            //Status da predição
            //            System.out.println("Status da predição" + r.get(i).getPredictStatus());
            //            //Valor dos inputs
            //            System.out.println("Valores do input: " + r.get(0).getInputsValuesReport().get(0)[0]);
            //            System.out.println("Valores do input: " + r.get(1).getInputsValuesReport().get(0)[1]);
            //            System.out.println("Valores do input: " + r.get(0).getInputsValuesReport().get(1)[0]);
            //            System.out.println("Valores do input: " + r.get(1).getInputsValuesReport().get(1)[1]);
            //            //Valor dos pesos
            //            System.out.println("Valores do peso inicial: " + r.get(0).getInitWeightsValuesReport().get(0));
            //            System.out.println("Valores do peso inicial: " + r.get(0).getInitWeightsValuesReport().get(1));
            //            //Valor dos delta pesos
            //            System.out.println("Valores do delta pesos: " + r.get(1).getDeltaWeightsValues().get(0));
            //            System.out.println("Valores do delta pesos: " + r.get(1).getDeltaWeightsValues().get(1));
            //            //Valor dos novos pesos
            //            System.out.println("Valores do novo peso: " + r.get(1).getNewWeightsValues().get(0));
            //            System.out.println("Valores do novo peso: " + r.get(1).getNewWeightsValues().get(1));
            //            //Valor da somatória
            //            System.out.println("Valor da somatória: " + r.get(i).getSumValue());
            //            //Função de ativação
            //            System.out.println("Função de ativação: " + r.get(i).getFunctionActivaion().name());
            //            //Valor do erro
            //            System.out.println("Valor do erro: " + r.get(i).getErrorValue());
            //            //Valor do delta bias
            //            System.out.println("Valor do delta bias: " + r.get(i).getDeltaBias());
            //            //Valor do novo bias
            //            System.out.println("Valor do novo bias: " + r.get(i).getNewBias());


        }

        return nn_content;
    }


    //Todo Jean - Desenvoler o front nessa função
    public static void report(ArrayList<Perceptron> r, String report_name) {
        String content = generateReportContentHTMl(r);
        String report_teste = Report.generateHTML(report_name, content);

        Report.reportPrint(report_name, report_teste);
    }
}
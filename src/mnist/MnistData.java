package mnist;

public class MnistData {

    private int[][] dataMatrix;
    private int[] dataArray;

    private int nRows;
    private int nCols;
    private int nSize;

    private int label;

    public MnistData(int nRows, int nCols) {
        this.nRows = nRows;
        this.nCols = nCols;

        dataMatrix = new int[nRows][nCols];
    }

    public MnistData(int size) {
        this.nSize = size;
        dataArray = new int[size];
    }

    //Matrix
    public int getValueOfMatrix(int r, int c) {
        return dataMatrix[r][c];
    }

    public void setValueOfMatrix(int row, int col, int value) {
        dataMatrix[row][col] = value;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public int getNumberOfRows() {
        return nRows;
    }

    public int getNumberOfColumns() {
        return nCols;
    }

    //Array
    public int getValueOfArray(int size) {
        return dataArray[size];
    }

    public void setValueOfArray(int size, int value) {
        dataArray[size] = value;
    }

    public int getSizeOfArray() {
        return nSize;
    }


}

package triplet.utils;

import java.io.*;
import java.util.Scanner;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * IO operations for INDArrays, and memory-conscious store of INDArrays
 */
public class ArrayIO {

    /**
     * Type of the data in the array: 'f' for Float, and 'b' for Byte.
     */
    protected char dataType;

    /**
     * Shape of the array, saved in an array of integers.
     * The length of <code>shape</code> is the number of dimensions in the array.
     */
    protected int[] shape;

    /**
     * Strides of the array, saved in an array of integers.
     * The length of <code>strides</code> is the number of dimensions in the array.
     */
    protected int[] strides;

    /**
     * Container for the data.
     */
    protected Number[] data;

    /**
     * Constructs an <code>ArrayIO</code> from an array of <code>Numbers</code>.
     *
     * @param data array of <code>Numbers</code>
     * @param dataType type of the data: can be 'f' for Float or 'b' for Byte
     * @param arrayShape shape of the array, as an array of <code>int</code>. The length of the array sets
     *                   the number of dimensions of the array.
     */
    public ArrayIO(Number[] data, char dataType, int[] arrayShape) {
        this.dataType = dataType;
        this.shape = arrayShape.clone();
        this.data = data.clone();
        this.strides = calculateStrides(this.shape);
    }

    /**
     * Constructs an <code>ArrayIO</code> from an <code>INDArray</code>.
     *
     * The type of the array is currently always set to 'f' (Float).
     *
     * @param array INDArray to save as ArrayIO. The <code>ArrayIO</code> will have the same shape as <code>array</code>.
     */
    public ArrayIO(INDArray array) {
        this.dataType = 'f';
        this.shape = array.shape();
        this.strides = calculateStrides(this.shape);

        // Ensure the input array is in C order (i.e. row-first)
        array.setOrder('c');

        // Copy the data element by element
        this.data = new Float[array.length()];
        for (int n = 0; n < array.length(); n++)
            this.data[n] = array.getFloat(n);
    }

    /**
     * Calculates the strides of an array from its shape.
     *
     * @param shape array of integers with the shape of an array.
     * @return an array of integers with the strides. The result has the same length as the argument <code>shape</code>.
     */
    protected static int[] calculateStrides(int[] shape) {
        int[] strides = new int[shape.length];

        strides[strides.length-1] = 1;
        for (int dim = strides.length-2; dim >= 0; dim--)
            strides[dim] = strides[dim+1] * shape[dim+1];

        return strides;
    }

    /**
     * Returns the <code>ArrayIO</code> as an <code>INDArray</code>.
     *
     * @return an <code>INDArray</code> with the same data and shape. The output INDArray is always float (due to Nd4j).
     */
    public INDArray toINDArray() {
        INDArray ret = Nd4j.create(this.data.length);
        for (int n = 0; n < this.data.length; n++)
            ret.putScalar(n, this.data[n].floatValue());
        ret = ret.reshape(this.shape);
        return ret;
    }

    /**
     * Returns an sub-array of this array, as an <code>INDArray</code>.
     *
     * @param index one or several integers, indexing this array in one or several dimensions.
     * @return an <code>INDArray</code> with a number of dimensions equal to the number of dimensions of this array
     *         minus the number of indexes in argument.
     */
    public INDArray get(int... index) {
        INDArray ret;

        // Calculate the number of dimensions of the output
        int numDimensionsSlice = index.length;
        int numDimensionsData = this.shape.length;
        int numDimensionsOutput = numDimensionsData - numDimensionsSlice;

        // Calculate the index (in this.data) of the first element that we want to retrieve
        int startIndex = 0;
        for (int dim = 0; dim < index.length; dim++)
            startIndex += index[dim] * this.strides[dim];

        if (numDimensionsOutput == 0) {
            // Handle scalar output
            ret = Nd4j.create(1);
            ret.putScalar(0, this.data[startIndex].floatValue());

        } else if (numDimensionsOutput > 0) {
            // Handle array output
            // Create an empty array
            int[] outputShape = new int[numDimensionsOutput];
            for (int dim = 0; dim < numDimensionsOutput; dim++) {
                outputShape[dim] = this.shape[dim + numDimensionsSlice];
            }
            int outputSize = Misc.product(outputShape);
            ret = Nd4j.create(outputSize);

            // Fill the array with data
            for (int pos = 0; pos < outputSize; pos++)
                ret.putScalar(pos, this.data[startIndex+pos].floatValue());

            ret = ret.reshape(outputShape);

        } else {
            throw new IllegalArgumentException("Passed indexes have too many dimensions.");
        }
        return ret;
    }

    /**
     * Reads an array from a file.
     *
     * The file is a text file with the following format:
     * - The first line contains the data type (one char, <code>f</code> for float or <code>b</code> for bytes).
     * - The second line contains the rank, i.e. the number of dimensions of the array.
     * - The third line contains the length of each dimension of the array, as integers separated by spaces.
     * - The fourth line contains all the data on one line, assuming a row-first format. Numbers are separated by spaces.
     *
     * @param filePath path of the file with the data.
     * @return the <code>ArrayIO</code> created from the file.
     * @throws IOException
     */
    public static ArrayIO fromFile(String filePath) throws IOException {
        // Read the data type, rank, and shape
        FileReader in = new FileReader(filePath);
        Scanner scanner = new Scanner(in);
        char tmpDataType = scanner.nextLine().trim().charAt(0);
        int tmpRank = scanner.nextInt();
        int[] tmpSize = new int[tmpRank];
        for (int n = 0; n < tmpRank; n++) {
            tmpSize[n] = scanner.nextInt();
        }

        // Calculate the total size of the array
        int tmpTotalSize = Misc.product(tmpSize);

        // Read the data
        Number[] data;
        switch (tmpDataType) {
            case 'f':
                data = new Float[tmpTotalSize];
                for (int pos = 0; pos < tmpTotalSize; pos++)
                    data[pos] = scanner.nextFloat();
                break;
            case 'b':
                data = new Byte[tmpTotalSize];
                for (int pos = 0; pos < tmpTotalSize; pos++)
                    data[pos] = scanner.nextByte();
                break;
            default:
                throw new IllegalArgumentException("Unknown data type.");
        }

        scanner.close();
        in.close();
        return new ArrayIO(data, tmpDataType, tmpSize);
    }

    /**
     * Saves this array to a file.
     *
     * The file is a text file with the following format:
     * - The first line contains the data type (one char, <code>f</code> for float or <code>b</code> for bytes).
     * - The second line contains the rank, i.e. the number of dimensions of the array.
     * - The third line contains the length of each dimension of the array, as integers separated by spaces.
     * - The fourth line contains all the data on one line, assuming a row-first format. Numbers are separated by spaces.
     *
     * @param filePath path of the file where the array must be written.
     * @throws IOException
     */
    public void toFile(String filePath) throws IOException {

        Writer out = new FileWriter(filePath);

        // Write format information: the data type, rank, and the lengths along each dimension
        out.write(this.dataType);
        out.write('\n');
        out.write(this.shape.length + "\n");
        for (int dim: this.shape)
            out.write(dim + " ");
        out.write('\n');

        // Write the data
        for (Number value: this.data)
            out.write(value.toString() + " ");

        out.flush();
        out.close();
    }

    /**
     * Returns the shape of this array.
     *
     * @return an array of int with the shape.
     */
    public int[] getShape() {
        return shape.clone();
    }
}

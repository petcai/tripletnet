package triplet.dataset;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import triplet.utils.ArrayIO;
import triplet.utils.Misc;

/**
 * A sampler of MNIST digits that yields triplets of images.
 *
 * In each triplet, the first and second element are of the same class, and the third is of a different class.
 */
public class MnistTripletSampler {

    /**
     * Number of different classes to load. If lower than 10, only the first <code>numLabels</code> digits are loaded.
     */
    protected int numLabels = 3;

    /**
     * Array containing all training images, taking up one byte by pixel.
     */
    protected ArrayIO[] mnistDB;

    /**
     * Height and width of an image side.
     */
    protected int imgSize = 28;

    /**
     * Mean and standard deviation of training images, used for normalizing inputs.
     */
    INDArray avgImage, stdImage;

    /**
     * Constructs a MNIST triplets sampler.
     *
     * @param mnistDbPath path of the folder with MNIST training files
     * @throws IOException
     */
    public MnistTripletSampler(String mnistDbPath) throws IOException {

        // Initialize variables
        mnistDB = new ArrayIO[numLabels];
        Path mnistDbPathObject = Paths.get(mnistDbPath);
        String filePath;

        // Load the training images, class by class
        for (int label = 0; label < numLabels; label++) {
            filePath = mnistDbPathObject.resolve("train_images" + label + ".txt").toString();
            mnistDB[label] = ArrayIO.fromFile(filePath);
        }

        // Load the normalization files
        filePath = mnistDbPathObject.resolve("train_images_mean.txt").toString();
        avgImage = ArrayIO.fromFile(filePath).toINDArray().reshape(1, imgSize * imgSize);
        filePath = mnistDbPathObject.resolve("train_images_std.txt").toString();
        stdImage = ArrayIO.fromFile(filePath).toINDArray().reshape(1, imgSize * imgSize);
    }

    /**
     * Generates random batches of image triplets.
     *
     * The batch contains <code>batchSize</code> triplets, thus 3 x <code>batchSize</code> images.
     * In each triplet, the first and second element are of the same label, and the third is of a different label.
     * Labels are drawn independently for each batch element.
     *
     * @param rng random number generator to draw image labels and images
     * @param batchSize the number of triplet to draw
     * @return an length-3 array of INDArrays. The n-th INDArray (n=1-3) contains the n-th element for all triplets.
     *      Each INDArray has size <code>batchSize x 1 x 28 x 28</code>.
     */
    public INDArray[] getTripletSample(Random rng, int batchSize) {
        // Initialize variables
        int[] labels;
        int[] imgPosIdx;
        int imgNegIdx, numImagesInClass;
        INDArray[] ret = new INDArray[3];
        for (int n=0; n < 3; n++) {
            ret[n] = Nd4j.create(new int[]{batchSize, imgSize*imgSize});
        }

        // For each record in the batch
        for (int n = 0; n < batchSize; n++) {
            // Sample a pair of labels
            labels = Misc.randomPair(rng, numLabels);

            // Find 2 images from the first label
            numImagesInClass = mnistDB[labels[0]].getShape()[0];
            imgPosIdx = Misc.randomPair(rng, numImagesInClass);
            ret[0].putRow(n, getImage(labels[0], imgPosIdx[0]));
            ret[1].putRow(n, getImage(labels[0], imgPosIdx[1]));

            // Find 1 image of the second label
            numImagesInClass = mnistDB[labels[1]].getShape()[0];
            imgNegIdx = rng.nextInt(numImagesInClass);
            ret[2].putRow(n, getImage(labels[1], imgNegIdx));
        }

        // Normalize the images and set the shape of the output
        for (int n=0; n < 3; n++) {
            ret[n].addi(128);
            ret[n].subiRowVector(avgImage);
            ret[n].diviRowVector(stdImage);
            ret[n] = ret[n].reshape(new int[]{batchSize, 1, imgSize, imgSize});
        }
        return ret;
    }

    /**
     * Retrieves a given image from a given label.
     *
     * @param label label of the image
     * @param idx index of the image in the data file
     * @return an 1-d array of length <code>28 * 28</code>.
     */
    // Attention: no checks performed on label and index
    public INDArray getImage(int label, int idx) {
        return mnistDB[label].get(idx);
    }

}

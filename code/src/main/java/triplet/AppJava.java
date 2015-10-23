package triplet;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import triplet.dataset.MnistTripletSampler;
import triplet.utils.ArrayIO;
import triplet.utils.WeightsIO;

/**
 * Training and execution of a triplet neural network
 */
public class AppJava {

    /**
     * Object to read the configuration of the program
     */
    private static Properties properties;

    /**
     * Triplet neural network model
     */
    private static TripletNetwork model;

    /**
     * Random number generator
     */
    private static Random rng;

    /**
     * Seed of the random number generator
     */
    private static int seed = 123;

    /**
     * Number of images in a training or testing batch
     */
    private static int batchSize;

    /**
     * Number of batches to use for training
     */
    private static long numBatches;

    /**
     * Width / height of an image
     */
    private static int imgSize = 28;

    /**
     * Logger
     */
    private static final Logger log = LoggerFactory.getLogger(AppJava.class);

    /**
     * Runs a triplet network, for training or testing, depending on the configuration file.
     *
     * @param args path of the configuration file
     */
    public static void main(String[] args) {
        rng = new Random(seed);

        try {
            // Read the properties file
            String propertiesPath = args[0];
            properties = readProperties(propertiesPath);

            // Configure the model
            log.info("Configure model...");
            configureModel();

            // Train or run it
            if (properties.getProperty("runType").equals("train")) {
                log.info("Train model...");
                trainModel();
            }
            else if (properties.getProperty("runType").equals("run")) {
                log.info("Run model...");
                runModel();
            }
            else {
                throw new IllegalArgumentException(
                        "Wrong configuration: runType " + properties.getProperty("runType") + " is not valid.");
            }

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException();
        }
    }

    /**
     * Configures and initializes the triplet network model
     *
     * @throws IOException
     */
    public static void configureModel() throws IOException {

        // Set parameters from the configuration file
        int numDescentIterations = Integer.parseInt(properties.getProperty("numDescentIterations"));
        int descentLoggingFrequency = Integer.parseInt(properties.getProperty("descentLoggingFrequency"));
        float learningRate = Float.parseFloat(properties.getProperty("learningRate"));
        float regularizationCoeff = Float.parseFloat(properties.getProperty("regularizationCoeff"));
        float momentum = Float.parseFloat(properties.getProperty("momentum"));
        boolean flagInitFromInputWeights = Boolean.parseBoolean(properties.getProperty("flagInitFromInputWeights"));
        String pathInputWeights = properties.getProperty("pathInputWeights");
        batchSize = Integer.parseInt(properties.getProperty("batchSize"));

        // Configure the model
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .batchSize(batchSize)
                .iterations(numDescentIterations)
                .regularization(true).l2(regularizationCoeff).momentum(momentum)
                .useDropConnect(true).learningRate(learningRate)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(7)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(1)
                        .nOut(4).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{3, 3})
                        .stride(3, 3).build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(16).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2})
                        .build())
                .layer(4, new ConvolutionLayer.Builder(2, 2)
                        .nOut(32).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2})
                        .build())
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.CUSTOM)
                        .nOut(32)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder, imgSize, imgSize, 1);
        MultiLayerConfiguration conf = builder.build();

        // Build the model from the configuration object
        model = new TripletNetwork(conf);
        model.init();
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(descentLoggingFrequency)));

        // Set the network weights from an input file
        if (flagInitFromInputWeights) {
            Map<String, INDArray> weightsMap = WeightsIO.fromFiles(pathInputWeights);
            for (Map.Entry<String, INDArray> entry : weightsMap.entrySet()) {
                model.setParam(entry.getKey(), entry.getValue());
            }
        }
    }

    /**
     * Trains the model
     *
     * @throws IOException
     */
    public static void trainModel() throws IOException {
        // Set parameters from the configuration file
        String pathMnist = properties.getProperty("pathMnist");
        numBatches = Long.parseLong(properties.getProperty("numBatches"));
        long loggingFrequency = Long.parseLong(properties.getProperty("weightsLoggingFrequency"));
        String pathWeightsLog = properties.getProperty("pathWeightsLog");
        String pathOutputWeights = properties.getProperty("pathOutputWeights");
        long timeBeforeLogging = loggingFrequency;
        INDArray[] batch;

        // Load the MNIST training database
        MnistTripletSampler trainDb = new MnistTripletSampler(pathMnist);

        for(long numBatch=0; numBatch < numBatches; numBatch++) {
            // Load a batch
            batch = trainDb.getTripletSample(rng, batchSize);

            // Train the model
            model.fit(batch);

            // Logging: Save the weights to file at regular intervals.
            timeBeforeLogging--;
            if(timeBeforeLogging <= 0) {
                Map<String, INDArray> weights = model.paramTable();
                try {
                    // Create a folder
                    Path logFolderPath = Paths.get(pathWeightsLog).resolve("batch_"+numBatch);
                    File logFolder = new File(logFolderPath.toString());
                    logFolder.mkdirs();

                    // Save the weights
                    WeightsIO.toFiles(weights, logFolderPath.toString());

                } catch (IOException e) {
                    // If failure to log, print the stacktrace but continue running.
                    e.printStackTrace();
                }

                // Reset the time before the next logging action.
                timeBeforeLogging = loggingFrequency;
            }
        }

        // Save final network weights after training
        Map<String, INDArray> weights = model.paramTable();
        WeightsIO.toFiles(weights, pathOutputWeights);
    }

    /**
     * Runs the model on images
     *
     * @throws IOException
     */
    public static void runModel() throws IOException {
        // Set parameters from the configuration file
        String pathInputImages = properties.getProperty("pathInputImages");
        String pathOutputImages = properties.getProperty("pathOutputImages");
        String pathNormalization = properties.getProperty("pathMnist");
        Path pathNormalizationObject = Paths.get(pathNormalization);
        INDArray avgImage, stdImage, input;

        // Load the coefficients that normalize inputs
        avgImage = ArrayIO.fromFile(pathNormalizationObject.resolve("train_images_mean.txt").toString())
                .toINDArray().reshape(1, imgSize*imgSize);
        stdImage = ArrayIO.fromFile(pathNormalizationObject.resolve("train_images_std.txt").toString())
                .toINDArray().reshape(1, imgSize*imgSize);

        // Load the input dataset and normalize it
        // NB: temporarily put the input in 2 dimensions to use broadcasting (not clear if ND4J broadcasts in 3+ dimensions)
        input = ArrayIO.fromFile(pathInputImages).toINDArray();
        input = input.reshape(input.shape()[0], imgSize*imgSize);
        // Images are saved with type "signed Byte", with values ranging from -128 to 127.
        // A O pixel value is saved as -128. Therefore I add 128 below.
        input.addi(128);
        input.subiRowVector(avgImage);
        input.diviRowVector(stdImage);
        input = input.reshape(input.shape()[0], 1, imgSize, imgSize);

        // Feed the input through the network, batch by batch
        List<INDArray> batchOutputs = new LinkedList<INDArray>();
        int range = batchSize;
        for (int idx = 0; idx < input.shape()[0]; idx += range) {
            INDArray batchInput = input.get(
                    NDArrayIndex.interval(idx, range + idx), NDArrayIndex.all(), NDArrayIndex.all());
            batchOutputs.add(model.output(batchInput , false));
        }

        // Concatenate outputs and write to file
        ArrayIO output = new ArrayIO(
                Nd4j.toFlattened('c', batchOutputs).reshape(input.shape()[0], batchOutputs.get(0).shape()[1]));
        output.toFile(pathOutputImages);
    }

    /**
     * Reads the configuration of this app in a properties file
     * @param propertiesPath path of the properties file
     * @return a <code>Properties</code> object containing all properties required by the run
     * @throws IOException
     */
    private static Properties readProperties(String propertiesPath) throws IOException {
        InputStream input = null;
        Properties properties = new Properties();

        // Read properties from file
        input = new FileInputStream(propertiesPath);
        properties.load(input);
        input.close();

        return properties;
    }
}


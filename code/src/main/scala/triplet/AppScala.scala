package triplet

import java.io._
import java.nio.file.{Path, Paths}
import java.util._

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}
import triplet.dataset.MnistTripletSampler
import triplet.utils.{ArrayIO, WeightsIO}

/**
 * Training and execution of a triplet neural network
 */
object AppScala {
    /**
     * Object to read the configuration of the program
     */
    private var properties: Properties = null

    /**
     * Triplet neural network model
     */
    private var model: TripletNetwork = null

    /**
     * Random number generator
     */
    private var rng: Random = null

    /**
     * Seed of the random number generator
     */
    private var seed: Int = 123

    /**
     * Number of images in a training or testing batch
     */
    private var batchSize: Int = 0

    /**
     * Number of batches to use for training
     */
    private var numBatches: Long = 0L

    /**
     * Width / height of an image
     */
    private var imgSize: Int = 28

    /**
     * Logger
     */
    private val log: Logger = LoggerFactory.getLogger(this.getClass)

    /**
     * Runs a triplet network, for training or testing, depending on the configuration file.
     *
     * @param args path of the configuration file
     */
    def main(args: Array[String]) {
        rng = new Random(seed)
        try {
            // Read the properties file
            val propertiesPath: String = args(0)
            properties = readProperties(propertiesPath)

            // Configure the model
            log.info("Configure model...")
            configureModel

            // Train or run it
            if (properties.getProperty("runType") == "train") {
                log.info("Train model...")
                trainModel
            }
            else if (properties.getProperty("runType") == "run") {
                log.info("Run model...")
                runModel
            }
            else {
                throw new IllegalArgumentException("Wrong configuration: runType " + properties.getProperty("runType") + " is not valid.")
            }
        }
        catch {
            case e: Exception => {
                e.printStackTrace
                throw new RuntimeException
            }
        }
    }

    /**
     * Configures and initializes the triplet network model
     *
     * @throws IOException
     */
    @throws(classOf[IOException])
    def configureModel {
        // Set parameters from the configuration file
        val numDescentIterations: Int = properties.getProperty("numDescentIterations").toInt
        val descentLoggingFrequency: Int = properties.getProperty("descentLoggingFrequency").toInt
        val learningRate: Float = properties.getProperty("learningRate").toFloat
        val regularizationCoeff: Float = properties.getProperty("regularizationCoeff").toFloat
        val momentum: Float = properties.getProperty("momentum").toFloat
        val flagInitFromInputWeights: Boolean = properties.getProperty("flagInitFromInputWeights").toBoolean
        val pathInputWeights: String = properties.getProperty("pathInputWeights")
        batchSize = properties.getProperty("batchSize").toInt

        // Configure the model
        val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .batchSize(batchSize)
                .iterations(numDescentIterations)
                .regularization(true).l2(regularizationCoeff).momentum(momentum)
                .useDropConnect(true).learningRate(learningRate)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(7)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(1).nOut(4)
                        .dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build)
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](3, 3))
                        .stride(3, 3)
                        .build)
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(16)
                        .dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build)
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
                        .build)
                .layer(4, new ConvolutionLayer.Builder(2, 2)
                        .nOut(32)
                        .dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build)
                .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
                        .build)
                .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.CUSTOM)
                        .nOut(32)
                        .weightInit(WeightInit.XAVIER)
                        .build)
                .backprop(true)
                .pretrain(false)
        new ConvolutionLayerSetup(builder, imgSize, imgSize, 1)
        val conf: MultiLayerConfiguration = builder.build

        // Build the model from the configuration object
        model = new TripletNetwork(conf)
        model.init
        model.setListeners(Arrays.asList(new ScoreIterationListener(descentLoggingFrequency).asInstanceOf[IterationListener]))

        // Set the network weights from an input file
        if (flagInitFromInputWeights) {
            val weightsMap: Map[String, INDArray] = WeightsIO.fromFiles(pathInputWeights)
            import scala.collection.JavaConversions._
            for (entry <- weightsMap.entrySet) {
                model.setParam(entry.getKey, entry.getValue)
            }
        }
    }

    /**
     * Trains the model
     *
     * @throws IOException
     */
    @throws(classOf[IOException])
    def trainModel {
        // Set parameters from the configuration file
        val pathMnist: String = properties.getProperty("pathMnist")
        numBatches = properties.getProperty("numBatches").toLong
        val loggingFrequency: Long = properties.getProperty("weightsLoggingFrequency").toLong
        val pathWeightsLog: String = properties.getProperty("pathWeightsLog")
        val pathOutputWeights: String = properties.getProperty("pathOutputWeights")
        var timeBeforeLogging: Long = loggingFrequency
        var batch: Array[INDArray] = null

        // Load the MNIST training database
        val trainDb: MnistTripletSampler = new MnistTripletSampler(pathMnist)
        var numBatch = 0L
        for (numBatch <- 0L until numBatches) {
            // Load a batch
            batch = trainDb.getTripletSample(rng, batchSize)

            // Train the model
            model.fit(batch)

            // Logging: Save the weights to file at regular intervals.
            timeBeforeLogging -= 1
            if (timeBeforeLogging <= 0) {
                val weights: Map[String, INDArray] = model.paramTable
                try {
                    // Create a folder
                    val logFolderPath: Path = Paths.get(pathWeightsLog).resolve("batch_" + numBatch)
                    val logFolder: File = new File(logFolderPath.toString)
                    logFolder.mkdirs

                    // Save the weights
                    WeightsIO.toFiles(weights, logFolderPath.toString)
                }
                catch {
                    // If failure to log, print the stacktrace but continue running.
                    case e: IOException => {
                        e.printStackTrace
                    }
                }
                // Reset the time before the next logging action.
                timeBeforeLogging = loggingFrequency
            }
        }
        // Save final network weights after training
        val weights: Map[String, INDArray] = model.paramTable
        WeightsIO.toFiles(weights, pathOutputWeights)

    }

    /**
     * Runs the model on images
     *
     * @throws IOException
     */
    @throws(classOf[IOException])
    def runModel {
        // Set parameters from the configuration file
        val pathInputImages: String = properties.getProperty("pathInputImages")
        val pathOutputImages: String = properties.getProperty("pathOutputImages")
        val pathNormalization: String = properties.getProperty("pathMnist")
        val pathNormalizationObject: Path = Paths.get(pathNormalization)
        var avgImage: INDArray = null
        var stdImage: INDArray = null
        var input: INDArray = null

        // Load the coefficients that normalize inputs
        avgImage = ArrayIO.fromFile(pathNormalizationObject.resolve("train_images_mean.txt").toString).toINDArray.reshape(1, imgSize * imgSize)
        stdImage = ArrayIO.fromFile(pathNormalizationObject.resolve("train_images_std.txt").toString).toINDArray.reshape(1, imgSize * imgSize)

        // Load the input dataset and normalize it
        // NB: temporarily put the input in 2 dimensions to use broadcasting (not clear if ND4J broadcasts in 3+ dimensions)
        input = ArrayIO.fromFile(pathInputImages).toINDArray
        input = input.reshape(input.shape()(0), imgSize * imgSize)
        // Images are saved with type "signed Byte", with values ranging from -128 to 127.
        // A O pixel value is saved as -128. Therefore I add 128 below.
        input.addi(128)
        input.subiRowVector(avgImage)
        input.diviRowVector(stdImage)
        input = input.reshape(input.shape()(0), 1, imgSize, imgSize)

        // Feed the input through the network, batch by batch
        val batchOutputs: List[INDArray] = new LinkedList[INDArray]
        val range: Int = batchSize

        var idx = 0
        while (idx < input.shape()(0)) {
            val batchInput: INDArray = input.get(NDArrayIndex.interval(idx, range + idx), NDArrayIndex.all, NDArrayIndex.all)
            batchOutputs.add(model.output(batchInput, false))
            idx += range
        }

        // Concatenate outputs and write to file
        val output: ArrayIO = new ArrayIO(Nd4j.toFlattened('c', batchOutputs).reshape(input.shape()(0), batchOutputs.get(0).shape()(1)))
        output.toFile(pathOutputImages)
    }

    /**
     * Reads the configuration of this app in a properties file
     * @param propertiesPath path of the properties file
     * @return a <code>Properties</code> object containing all properties required by the run
     * @throws IOException
     */
    @throws(classOf[IOException])
    private def readProperties(propertiesPath: String): Properties = {
        var input: InputStream = null
        val properties: Properties = new Properties

        // Read properties from file
        input = new FileInputStream(propertiesPath)
        properties.load(input)
        input.close

        return properties
    }
}
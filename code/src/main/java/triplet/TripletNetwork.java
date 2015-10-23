package triplet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.optimize.Solver;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.api.ops.impl.transforms.Pow;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.functions.Value;
import org.nd4j.linalg.api.ndarray.INDArray;
import triplet.factories.LayerFactories;
import triplet.layers.TripletConvolutionLayer;
import triplet.layers.TripletOutputLayer;
import triplet.layers.TripletSubsamplingLayer;

/**
 * Neural network to train with triplets images.
 *
 * In each triplet, the first and second elements must correspond to the same label, and the third to a different label.
 * The network learns by calculating outputs for each image in the triplet, and
 * adapting in order to have closer outputs for images with the same label than for images with different labels.
 */
public class TripletNetwork extends org.deeplearning4j.nn.multilayer.MultiLayerNetwork {

    /**
     * Map of the state of each layer.
     *
     * This map stores the values of all activations of the feedforward pass of each triplet element.
     * Its purpose is to be able to restore the activation values before each of the three backpropagations.
     */
    protected HashMap<String, INDArray>[][] layerStates;

    /**
     * Length-3 array containing the 3 batches of input images.
     */
    protected INDArray[] tripletInput;

    /**
     * Map to contain the gradients for the triplet, corresponding to the sum of gradients for each triplet element.
     */
    protected Map<String, INDArray> tripletGradient;

    /**
     * Constructs a triplet network from the configuration.
     *
     * @param conf configuration of a MultiLayerNetwork
     */
    public TripletNetwork(MultiLayerConfiguration conf) {
        super(conf);
    }

    /**
     * Intializes the network.
     *
     * @author Copied from Deeplearning4j.
     */
    @Override
    public void init() {
        // Configure layers
        if (layerWiseConfigurations == null || layers == null)
            intializeConfigurations();
        if (initCalled)
            return;

        // Construct layers
        if (this.layers == null || this.layers[0] == null) {
            if (this.layers == null)
                this.layers = new Layer[getnLayers()];

            for (int i = 0; i < getnLayers(); i++) {
                NeuralNetConfiguration conf = layerWiseConfigurations.getConf(i);
                layers[i] = LayerFactories.getFactory(conf).create(conf,getListeners(), i);
            }
            initCalled = true;
            setMask(Nd4j.ones(1, pack().length()));
        }

        // Set parameters in MultiLayerNetwork.defaultConfiguration for later use in BaseOptimizer
        defaultConfiguration.clearVariables();
        for( int i=0; i<layers.length; i++ ){
            for( String s : layers[i].conf().variables() ){
                defaultConfiguration.addVariable(i+"_"+s);
            }
        }

        // Create a map to save layer states
        layerStates = new HashMap[3][getnLayers()];
        for (int n = 0; n < 3; n++) {
            for (int numLayer = 0; numLayer < getnLayers(); numLayer++)
                layerStates[n][numLayer] = new HashMap<String, INDArray>();
        }

    }

    /**
     * Saves the state of every layer.
     *
     * Depending on the layer type, its state can contain the activations, the inputs, the dropout mask,
     * the positions of maximum indexes (for max-pooling layers), etc.
     *
     * @param posMemory a number 0-2 identifying the triplet element corresponding to this state.
     */
    protected void saveLayerStates(int posMemory) {
        for (int numLayer = 0; numLayer < getnLayers(); numLayer++) {
            if (layers[numLayer] instanceof TripletConvolutionLayer)
                layerStates[posMemory][numLayer] = ((TripletConvolutionLayer) layers[numLayer]).saveState();
            if (layers[numLayer] instanceof TripletSubsamplingLayer)
                layerStates[posMemory][numLayer] = ((TripletSubsamplingLayer) layers[numLayer]).saveState();
            if (layers[numLayer] instanceof TripletOutputLayer)
                layerStates[posMemory][numLayer] = ((TripletOutputLayer) layers[numLayer]).saveState();
        }
    }

    /**
     * Restores the state of all layers
     *
     * Depending on the layer type, its state can contain the activations, the inputs, the dropout mask,
     * the positions of maximum indexes (for max-pooling layers), etc.
     *
     * @param posMemory a number 0-2 identifying the triplet element corresponding to this state.
     */
    protected void restoreLayerStates(int posMemory) {
        for (int numLayer = 0; numLayer < getnLayers(); numLayer++) {
            if (layers[numLayer] instanceof TripletConvolutionLayer)
                ((TripletConvolutionLayer) layers[numLayer]).restoreState(layerStates[posMemory][numLayer]);
            if (layers[numLayer] instanceof TripletSubsamplingLayer)
                ((TripletSubsamplingLayer) layers[numLayer]).restoreState(layerStates[posMemory][numLayer]);
            if (layers[numLayer] instanceof TripletOutputLayer)
                ((TripletOutputLayer) layers[numLayer]).restoreState(layerStates[posMemory][numLayer]);
        }
    }

    /**
     * Fits the model to a batch of triplets.
     *
     * @param data   the batch of triplets to train on.
     */
    public void fit(INDArray[] data) {
        setTripletInput(data);

        if( solver == null) {
            solver = new Solver.Builder()
                    .configure(conf())
                    .listeners(getListeners())
                    .model(this).build();
        }

        solver.optimize();
    }

    /**
     * Sets the triplet input.
     *
     * @param input   the batch of triplets to train on.
     */
    public void setTripletInput(INDArray[] input) {
        if (input.length != 3)
            throw new IllegalArgumentException("The input is not a triplet.");

        // Possibly: create a new INDArray, with duplicates of the three inputs
        tripletInput = input;
    }

    /**
     * Compute the gradient and the score of the network, based on the current batch of triplets.
     */
    @Override
    public void computeGradientAndScore() {
        List<INDArray> activations;
        List<INDArray> tripletOutput = new ArrayList<INDArray>();

        // Feedforward the three inputs through the network
        for (int n = 0; n < 3; n++) {
            setInput(tripletInput[n]);
            activations = feedForward();

            // Save the output, and the states of all layers
            tripletOutput.add(n, activations.get(activations.size()-1));
            saveLayerStates(n);
        }

        // Calculate the deltas (derivatives of the loss with respect to network outputs) and the score
        Pair<List<INDArray>, INDArray> deltasAndLoss = computeDeltasAndLoss(tripletOutput);
        List<INDArray> tripletOutputDeltas = deltasAndLoss.getFirst();
        INDArray loss = deltasAndLoss.getSecond();

        // Backpropagate separately for each triplet input
        tripletGradient = new HashMap<String, INDArray>();
        for (int n = 0; n < 3; n++) {
            restoreLayerStates(n);

            // Set the labels equal to the deltas: the TripletOutputLayer uses the delta as labels
            this.labels = tripletOutputDeltas.get(n);

            backprop();

            // Create a cumulated sum of gradients, which yields the gradient for the triplet network
            Map<String, INDArray> gradientMap = gradient().gradientForVariable();
            if (n == 0) {
                tripletGradient.putAll(gradientMap);
            } else {
                for (Map.Entry<String, INDArray> entry : gradientMap.entrySet()) {
                    tripletGradient.put(
                            entry.getKey(),
                            entry.getValue().add(tripletGradient.get(entry.getKey())));
                }
            }
        }

        // Set the network gradient attribute
        gradient = new DefaultGradient();
        for (Map.Entry<String, INDArray> entry : tripletGradient.entrySet()) {
            gradient.setGradientFor(entry.getKey(), entry.getValue());
        }

        // Calculate the score by averaging all losses in the minibatch and adding regularization terms
        score = loss.meanNumber().doubleValue() + calcL1() + calcL2();
    }

    /**
     * Computes the deltas of the output layer, and the loss corresponding to triplet outputs.
     *
     * @param tripletOutput a list of 3 INDArrays, corresponding to the batch of first, second, and third
     *                      triplet elements
     * @return a list of 3 INDArrays with deltas of the output layer, and an array of losses
     *      (one number by triplet in the batch).
     */
    protected Pair<List<INDArray>, INDArray> computeDeltasAndLoss(List<INDArray> tripletOutput) {
        /*
        Calculations and notations derive from the paper "Deep metric learning using Triplet network"
        by Elad Hoffer, Nir Ailon (http://arxiv.org/abs/1412.6622)

        diffOutputPos = Net(x) - Net(x^+)
        diffOutputNeg = Net(x) - Net(x^-)
        distOutputPos = || diffOutputPos ||_2
        distOutputNeg = || diffOutputNeg ||_2

        dPos = d_+

        scores = 2 * dPos^2 (loss)
        deltaOutputs = derivative of the loss function with respect to network outputs,
            (outputs being either Net(x), Net(x^+), or Net(x^-))
         */
        // Determine the batch size from the triplet outputs
        int batchSize = tripletOutput.get(0).shape()[0];

        // Calculate differences and distances between the triplet network outputs
        INDArray diffOutputPos = tripletOutput.get(0).sub(tripletOutput.get(1));
        INDArray diffOutputNeg = tripletOutput.get(0).sub(tripletOutput.get(2));
        INDArray distOutputPos = diffOutputPos.norm2(1);
        INDArray distOutputNeg = diffOutputNeg.norm2(1);
        INDArray deltaDistances = distOutputNeg.sub(distOutputPos);

        // Prevent distances equal to 0, which would cause a division by 0 (below)
        BooleanIndexing.applyWhere(distOutputPos, Conditions.lessThan(Nd4j.EPS_THRESHOLD), new Value(Nd4j.EPS_THRESHOLD));
        BooleanIndexing.applyWhere(distOutputNeg, Conditions.lessThan(Nd4j.EPS_THRESHOLD), new Value(Nd4j.EPS_THRESHOLD));

        // Prevent large values in the exponentials below
        BooleanIndexing.applyWhere(deltaDistances, Conditions.greaterThan(40.), new Value(40.));

        // Calculate the output deltas (derivatives of the loss with respect to network outputs)
        INDArray tmp = Nd4j.getExecutioner().execAndReturn(new Exp(deltaDistances.dup()));
        INDArray dPos = (tmp.add(1.)).rdiv(1.);
        INDArray tmp2 = Nd4j.getExecutioner().execAndReturn(new Pow(tmp.add(1.).dup(), 2));
        INDArray lossFactor = dPos.mul(tmp).mul(4.).div(tmp2);
        INDArray deltaPos = diffOutputPos.divColumnVector(distOutputPos.reshape(batchSize, 1)).neg().mulColumnVector(lossFactor.reshape(batchSize, 1));
        INDArray deltaNeg = diffOutputNeg.divColumnVector(distOutputNeg.reshape(batchSize, 1)).mulColumnVector(lossFactor.reshape(batchSize, 1));

        List<INDArray> deltaOutputs = new ArrayList<INDArray>(3);
        deltaOutputs.add(0, deltaPos.neg().sub(deltaNeg));
        deltaOutputs.add(1, deltaPos);
        deltaOutputs.add(2, deltaNeg);

        // Calculate the scores
        INDArray scores = Nd4j.getExecutioner().execAndReturn(new Pow(dPos, 2)).mul(2);

        Pair<List<INDArray>, INDArray> ret = new Pair<List<INDArray>, INDArray>(deltaOutputs, scores);
        return ret;

    }

    /**
     * Returns a 1 x m vector concatenating all weights and biases in the network.
     *
     * @return a 1-dimension INDArray with all parameters.
     */
    @Override
    public INDArray params() {

        List<INDArray> params = new ArrayList<INDArray>();
        for (int i = 0; i < getLayers().length-1; i++) {
            Layer layer = getLayer(i);
            params.add(layer.params());
        }

        return Nd4j.toFlattened('f',params);
    }

    /**
     * Sets the weights and biases of this model.
     *
     * @param params the parameters for the model concatenated in a 1-dimension array
     */
    @Override
    public void setParams(INDArray params) {
        int idx = 0;
        for (int i = 0; i < getLayers().length-1; i++) {
            Layer layer = getLayer(i);
            int range = layer.numParams();
            if (range > 0) {
                // Set the layer parameters equal to a range in the params vector
                INDArray get = params.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, range + idx));
                layer.setParams(get);
                idx += range;
            }
        }
    }
}

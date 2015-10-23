package triplet.layers;

import java.util.HashMap;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.OutputLayer;

/**
 * Output layer for triplet networks.
 *
 * This layer adds, on top of Nd4j's implementation, functions to save and restore the state of the layer.
 *
 * The ability to save and restore the state of the layer is useful for triplet networks. During training,
 * such networks run three feedforward passes, calculate the loss function, and then run three backpropagation passes.
 * The layer state must be saved after each feedforward pass, and restored before the corresponding backpropagation pass.
 *
 * However, this particular class has no variables that need to be saved between feedforward and backpropagation passes.
 * Nonetheless the save and restore functions are included to comply with an interface that could be used in other implementations.
 */
public class TripletOutputLayer extends OutputLayer {

    /**
     * Constructs the layer.
     *
     * @param conf configuration instructions for the layer
     */
    public TripletOutputLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    /**
     * Saves the state of the layer in a map.
     *
     * @return <code>null</code>, since this particular implementation has no state variables.
     */
    public HashMap<String, INDArray> saveState() {
        return null;
    }

    /**
     * Restores the state of the layer from a map.
     *
     * @param stateMap a map of state variables values, with variable names as keys.
     */
    public void restoreState(HashMap<String, INDArray> stateMap) {}

    /**
     * Sets the inputs of the layer.
     *
     * @param input the input of the output layer in an INDArray
     * @param training <code>true</code> during the training phase
     */
    @Override
    public void setInput(INDArray input, boolean training) {
        this.input = input;
    }

    /**
     * Returns the outputs of the layer.
     *
     * @param training <code>true</code> during the training phase
     * @return the outputs in an <code>INDArray</code>
     */
    @Override
    public  INDArray output(boolean training) {
        return activate(true);
    }

    /**
     * Returns the activations at the output of the layer.
     *
     * For this particular implementation, the outputs are equal to the inputs.
     *
     * @param training <code>true</code> during the training phase
     */
    @Override
    public INDArray activate(boolean training) {
        return input();
    }

    /**
     * Back-propagates the gradient.
     *
     * However, in this implementation the output layer has no weights, so the first output is
     * an empty <code>Gradient</code> object.
     *
     * @param epsilon epsilon of the next layer, which is null for the output layer.
     * @return an empty gradient, and the output deltas.
     */
    @Override
    public Pair<Gradient,INDArray> backpropGradient(INDArray epsilon) {

        // The gradient is empty because this output layer has no weights
        Gradient gradient = new DefaultGradient();

        // Implementation note:
        // The loss function does not directly use labels: instead it compares the outputs corresponding to a triplet of inputs.
        // Here, the labels attribute is used to save the derivative of this loss with respect to network outputs.
        // This ensures the compatibility with the backpropagation algorithm of the super class.
        return new Pair<Gradient,INDArray>(gradient, labels);
    }

}

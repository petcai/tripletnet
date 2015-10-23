package triplet.layers;

import java.util.HashMap;
import java.util.Arrays;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;

/**
 * Convolution layer for triplet networks.
 *
 * This layer adds, on top of Nd4j's implementation, functions to save and restore the state of the layer.
 *
 * The ability to save and restore the state of the layer is useful for triplet networks. During training,
 * such networks run three feedforward passes, calculate the loss function, and then run three backpropagation passes.
 * The layer state must be saved after each feedforward pass, and restored before the corresponding backpropagation pass.
 *
 * For this class, the state is <code>col</code> (a reshaped layer input) and <code>dropoutMask</code> INDArrays.
 */
public class TripletConvolutionLayer extends ConvolutionLayer {

    /**
     * Constructs the layer.
     *
     * @param conf configuration instructions for the layer
     */
    public TripletConvolutionLayer(NeuralNetConfiguration conf) {
        super(conf);
        col = null;
        gradient = null;
    }

    /**
     * Saves the state of the layer in a map.
     *
     * @return a map with variable names as keys, and copies of the state variables as values.
     */
    public HashMap<String, INDArray> saveState() {

        HashMap<String, INDArray> state = new HashMap<String, INDArray>();
        if (col != null)
            state.put("col", col.dup());
        else
            state.put("col", null);

        if (dropoutMask != null)
            state.put("dropoutMask", dropoutMask.dup());
        else
            state.put("dropoutMask", null);

        return state;
    }

    /**
     * Restores the state of the layer from a map.
     *
     * @param stateMap a map of state variables values, with variable names as keys.
     */
    public void restoreState(HashMap<String, INDArray> stateMap) {
        // Restore col
        if ((stateMap.get("col") != null) && (col != null)) {
            // Use the faster "assign" function if the arrays have the right shape
            if (Arrays.equals(col.shape(), stateMap.get("col").shape())) {
                col.assign(stateMap.get("col"));
            } else {
                col = stateMap.get("col").dup();
            }
        } else if (stateMap.get("col") != null) {
            col = stateMap.get("col").dup();
        } else {
            col = null;
        }

        // Restore dropoutMask
        if ((stateMap.get("dropoutMask") != null) && (dropoutMask != null)) {
            // Use the faster "assign" function if the arrays have the right shape
            if (Arrays.equals(dropoutMask.shape(), stateMap.get("dropoutMask").shape())) {
                dropoutMask.assign(stateMap.get("dropoutMask"));
            } else {
                dropoutMask = stateMap.get("dropoutMask").dup();
            }
        } else if (stateMap.get("dropoutMask") != null) {
            dropoutMask = stateMap.get("dropoutMask").dup();
        } else {
            dropoutMask = null;
        }
    }
}

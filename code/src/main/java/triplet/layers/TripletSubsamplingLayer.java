package triplet.layers;

import java.util.Arrays;
import java.util.HashMap;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.loop.coordinatefunction.CoordinateFunction;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.util.Dropout;


/**
 * Subsampling layer for triplet networks.
 *
 * This class is mostly copied from the SubsamplingLayer of deeplearning4j (extending the class was not an option
 * because we need access to the maxIndexes private variable).
 *
 * This layer adds, on top of Nd4j's implementation, functions to save and restore the state of the layer.
 *
 * The ability to save and restore the state of the layer is useful for triplet networks. During training,
 * such networks run three feedforward passes, calculate the loss function, and then run three backpropagation passes.
 * The layer state must be saved after each feedforward pass, and restored before the corresponding backpropagation pass.
 *
 * For this class, the state is <code>maxIndexes</code> (the positions of maximums in the pooling layer)
 * and <code>dropoutMask</code> INDArrays.
 *
 * @author Adam Gibson, Harold Sneessens
 */
public class TripletSubsamplingLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.SubsamplingLayer> {

    /**
     * Position of maximums in the pooling layer.
     */
    protected INDArray maxIndexes;

    /**
     * Constructs the layer.
     *
     * @param conf configuration instructions for the layer
     */
    public TripletSubsamplingLayer(NeuralNetConfiguration conf) {
        super(conf);
        maxIndexes = null;
        gradient = null;
    }

    /**
     * Saves the state of the layer in a map.
     *
     * @return a map with variable names as keys, and copies of the state variables as values.
     */
    public HashMap<String, INDArray> saveState() {
        HashMap<String, INDArray> state = new HashMap<String, INDArray>();

        if (maxIndexes != null)
            state.put("maxIndexes", maxIndexes.dup());
        else
            state.put("maxIndexes", null);

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
        // Restore maxIndexes
        if ((stateMap.get("maxIndexes") != null) && (maxIndexes != null)) {
            // Use the faster "assign" function if the arrays have the right shape
            if (Arrays.equals(maxIndexes.shape(), stateMap.get("maxIndexes").shape())) {
                maxIndexes.assign(stateMap.get("maxIndexes"));
            } else {
                maxIndexes = stateMap.get("maxIndexes").dup();
            }
        } else if (stateMap.get("maxIndexes") != null) {
            maxIndexes = stateMap.get("maxIndexes").dup();
        } else {
            maxIndexes = null;
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

    /**
     * Back-propagates the deltas/epsilon.
     *
     * @param epsilon epsilon from the previous layer, to backpropagate.
     * @return an empty gradient, and the epsilons for the next layer.
     */
    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        // Subsampling doesn't have weights and thus gradients are not calculated for this layer.
        // Only scale and reshape epsilon.
        int inputHeight = input().size(-2);
        int inputWidth = input().size(-1);
        INDArray reshapeEpsilon, retE, reshaped;
        Gradient retGradient = new DefaultGradient();

        switch(layerConf().getPoolingType()) {
            case MAX:
                int n = epsilon.size(0);
                int c = epsilon.size(1);
                int outH = epsilon.size(2);
                int outW = epsilon.size(3);

                // Compute backwards kernel based on rearranging the given error
                retE = Nd4j.zeros(n, c, layerConf().getKernelSize()[0], layerConf().getKernelSize()[1], outH, outW);
                reshaped = retE.reshape(n, c, -1, outH, outW);
                reshapeEpsilon = Nd4j.rollAxis(reshaped,2);
                final INDArray finalEps = epsilon;
                final INDArray reshapedEps = reshapeEpsilon;
                Shape.iterate(0, 4, new int[]{n, c, outH, outW}, new int[4], new CoordinateFunction() {
                    @Override
                    public void process(int[]... coord) {
                        try {
                            int[] i = coord[0];
                            double epsGet = finalEps.getDouble(i);
                            int idx = maxIndexes.getInt(i);
                            INDArray sliceToGetFrom = reshapedEps.get(NDArrayIndex.point(idx));
                            sliceToGetFrom.putScalar(i, epsGet);
                        } catch (Exception e) {
                            throw new IllegalStateException("Iterated to " + Arrays.toString(coord[0]) + " out of shape for indexes "
                                    + Arrays.toString(maxIndexes.shape()) + " and final eps shape " + Arrays.toString(finalEps.shape()));
                        }
                    }
                });

                reshapeEpsilon = Convolution.col2im(retE, layerConf().getStride(), layerConf().getPadding(), inputHeight, inputWidth);
                return new Pair<>(retGradient,reshapeEpsilon);
            case AVG:
                // Compute reverse average error
                retE = epsilon.get(
                        NDArrayIndex.all()
                        , NDArrayIndex.all()
                        , NDArrayIndex.newAxis()
                        , NDArrayIndex.newAxis());
                reshapeEpsilon = Nd4j.tile(retE,1,1,layerConf().getKernelSize()[0],layerConf().getKernelSize()[1],1,1);
                reshapeEpsilon = Convolution.col2im(reshapeEpsilon, layerConf().getStride(), layerConf().getPadding(), inputHeight, inputWidth);
                reshapeEpsilon.divi(ArrayUtil.prod(layerConf().getKernelSize()));

                return new Pair<>(retGradient, reshapeEpsilon);
            case NONE:
                return new Pair<>(retGradient, epsilon);
            default: throw new IllegalStateException("Unsupported pooling type");
        }
    }

    /**
     * Calculates activations of this layer.
     *
     * @param training <code>true</code> during the training phase
     * @return the activations
     */
    @Override
    public INDArray activate(boolean training) {
        INDArray pooled, ret;
        // n = num examples, c = num channels or depth
        int n, c, kh, kw, outWidth, outHeight;

        // Apply dropout
        if(training && conf.getLayer().getDropOut() > 0) {
            this.dropoutMask = Dropout.applyDropout(input, conf.getLayer().getDropOut(), dropoutMask);
        }

        // Calculate activations (max or avg)
        pooled = Convolution.im2col(input,layerConf().getKernelSize(),layerConf().getStride(),layerConf().getPadding());
        switch(layerConf().getPoolingType()) {
            case AVG:
                return pooled.mean(2,3);
            case MAX:
                n = pooled.size(0);
                c = pooled.size(1);
                kh = pooled.size(2);
                kw = pooled.size(3);
                outWidth = pooled.size(4);
                outHeight = pooled.size(5);
                ret = pooled.reshape(n, c, kh * kw, outHeight, outWidth);
                maxIndexes = Nd4j.argMax(ret, 2);
                return ret.max(2);
            case NONE:
                return input;
            default: throw new IllegalStateException("Pooling type not supported!");

        }
    }

    /**
     * Returns the L2 regularization term.
     *
     * @return 0 since the layer has no weights.
     */
    @Override
    public double calcL2() {
        return 0;
    }

    /**
     * Returns the L1 regularization term.
     *
     * @return 0 since the layer has no weights.
     */
    @Override
    public double calcL1() {
        return 0;
    }

    /**
     * Returns the type of the layer.
     *
     * @return the type
     */
    @Override
    public Type type() {
        return Type.SUBSAMPLING;
    }

    @Override
    public Gradient error(INDArray input) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray activationMean() {
        return Nd4j.create(0);
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void iterate(INDArray input) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit() {}

    @Override
    public int numParams() {
        return 0;
    }

    @Override
    public void fit(INDArray input) {}

    @Override
    public void computeGradientAndScore() {}

    @Override
    public double score() {
        return 0;
    }

    @Override
    public void accumulateScore(double accum) { throw new UnsupportedOperationException(); }


    @Override
    public void update(INDArray gradient, String paramType) {}

    @Override
    public INDArray params() {
        return Nd4j.create(0);
    }

    @Override
    public INDArray getParam(String param) {
        return params();
    }

    @Override
    public void setParams(INDArray params) {}

}

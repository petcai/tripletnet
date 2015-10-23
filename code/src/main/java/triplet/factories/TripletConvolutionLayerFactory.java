package triplet.factories;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;

/**
 * Factory for convolutional layers.
 *
 * Copied from deeplearning4j and modified.
 *
 * @author Adam Gibson
 */
public class TripletConvolutionLayerFactory extends DefaultTripletLayerFactory {
    public TripletConvolutionLayerFactory(Class<? extends Layer> layerConfig) {
        super(layerConfig);
    }

    @Override
    public ParamInitializer initializer() {
        return new ConvolutionParamInitializer();
    }
}

package triplet.factories;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.SubsampleParamInitializer;

/**
 * Factory for subsampling layers.
 *
 * Copied from deeplearning4j and modified.
 *
 * @author Adam Gibson
 */
public class TripletSubsampleLayerFactory extends DefaultTripletLayerFactory {
    public TripletSubsampleLayerFactory(Class<? extends Layer> layerClazz) {
        super(layerClazz);
    }

    @Override
    public ParamInitializer initializer() {
        return new SubsampleParamInitializer();
    }
}

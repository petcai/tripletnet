package triplet.factories;

import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;

/**
 * Static methods for finding which layer factory to use.
 *
 * Copied from deeplearning4j and modified, because of the impossibility to inject dependencies.
 *
 * @author Adam Gibson
 * @author Harold Sneessens
 */
public class LayerFactories {
    /**
     * Get the factory based on the passed in class
     *
     * @param conf the clazz to get the layer factory for
     * @return the layer factory for the particular layer
     */
    public static LayerFactory getFactory(NeuralNetConfiguration conf) {
        return getFactory(conf.getLayer());
    }

    /**
     * Get the factory based on the passed in class
     *
     * @param layer the clazz to get the layer factory for
     * @return the layer factory for the particular layer
     */
    public static LayerFactory getFactory(Layer layer) {
        Class<? extends Layer> clazz = layer.getClass();
        if(ConvolutionLayer.class.isAssignableFrom(clazz))
            return new TripletConvolutionLayerFactory(clazz);
        else if(SubsamplingLayer.class.isAssignableFrom(clazz))
            return new TripletSubsampleLayerFactory(clazz);
        return new DefaultTripletLayerFactory(clazz);
    }

    /**
     * Get the type for the layer factory
     *
     * @param conf the layer factory
     * @return the type
     */
    public static org.deeplearning4j.nn.api.Layer.Type typeForFactory(NeuralNetConfiguration conf) {
        LayerFactory layerFactory = getFactory(conf);
        if(layerFactory instanceof TripletConvolutionLayerFactory
                || layerFactory instanceof TripletSubsampleLayerFactory)
            return org.deeplearning4j.nn.api.Layer.Type.CONVOLUTIONAL;
        else if(layerFactory instanceof DefaultTripletLayerFactory)
            return org.deeplearning4j.nn.api.Layer.Type.FEED_FORWARD;

        throw new IllegalArgumentException("Unknown layer type");
    }

}

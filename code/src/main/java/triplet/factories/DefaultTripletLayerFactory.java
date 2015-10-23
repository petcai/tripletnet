package triplet.factories;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

/**
 * Default factory for layers.
 */
public class DefaultTripletLayerFactory extends org.deeplearning4j.nn.layers.factory.DefaultLayerFactory {

    /**
     * Builds the factory.
     *
     * @param layerConfig configuration of the layer
     */
    public DefaultTripletLayerFactory(Class<? extends org.deeplearning4j.nn.conf.layers.Layer> layerConfig) {
        super(layerConfig);
    }

    /**
     * Create a layer.
     *
     * @param conf configuration of the layer
     * @return a network layer. The class of the layer depends on the configuration.
     */
    protected Layer getInstance(NeuralNetConfiguration conf) {
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.OutputLayer)
            return new triplet.layers.TripletOutputLayer(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.ConvolutionLayer)
            return new triplet.layers.TripletConvolutionLayer(conf);
        if(layerConfig instanceof org.deeplearning4j.nn.conf.layers.SubsamplingLayer)
            return new triplet.layers.TripletSubsamplingLayer(conf);
        throw new RuntimeException("unknown layer type: " + layerConfig);
    }
}
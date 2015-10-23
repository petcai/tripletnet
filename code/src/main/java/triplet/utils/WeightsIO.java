package triplet.utils;

import java.util.*;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * IO operations for network weights.
 */
public class WeightsIO {

    /**
     * Writes a map of weights to files.
     *
     * Each weights array is saved in a file with the weight name (the key in the map) as name.
     * The file format is the one of <code>ArrayIO</code>.
     *
     * @param weightsMap the map of weights to save. Keys must be valid filenames.
     * @param folderPath the folder where the weights must be saved. Each weights array is saved in a different file.
     * @throws IOException
     */
    public static void toFiles(Map<String, INDArray> weightsMap, String folderPath) throws IOException {
        Path folderPathObject = Paths.get(folderPath);
        String filePath;

        // Write each weight / bias matrix in a different file, whose name is the key in the map.
        // For weights maps, the keys are of the form <layer number>_<W/b>.
        for (Map.Entry<String, INDArray> entry : weightsMap.entrySet()) {
            filePath = folderPathObject.resolve(entry.getKey()).toString();
            ArrayIO weightsArray = new ArrayIO(entry.getValue());
            weightsArray.toFile(filePath);
        }
    }

    /**
     *
     /**
     * Reads a map of weights from files.
     *
     * Each file in the folder is assumed to contain a valid weights array.
     * The file format is the one of <code>ArrayIO</code>.
     *
     * @param folderPath the folder where the weights are be saved. Each file must corresponds to a weights array.
     * @return a map with the weights. The map keys are set to the filenames on disk.
     * @throws IOException
     */
    public static Map<String, INDArray> fromFiles(String folderPath) throws IOException {
        Map<String, INDArray> ret = new HashMap<String, INDArray>();
        File ff = new File(folderPath);

        // Read all files in the folder and save them in a map, with the filename as key.
        File[] fileNames = ff.listFiles();
        for (int i = 0; i < fileNames.length; i++) {
            ret.put(fileNames[i].getName(), ArrayIO.fromFile(fileNames[i].getAbsolutePath()).toINDArray());
        }

        return ret;
    }
}

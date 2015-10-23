package triplet.utils;

import java.util.Random;

/**
 * Miscellaneous static mathematical functions.
 */
public class Misc {

    /**
     * Calculates the product of elements in an array.
     *
     * @param array an integer array.
     * @return the product of all elements in the array, as an integer.
     */
    public static int product(int[] array) {
        int ret = 1;
        for (int value: array)
            ret *= value;
        return ret;
    }

    /**
     * Calculates the cumulative product of elements in an array.
     *
     * @param array an integer array.
     * @return an integer array of the same size as the input array, containing in each position the product
     *         of all elements up to this position.
     */
    public static int[] cumulativeProduct(int[] array) {
        int[] ret = new int[array.length];

        ret[0] = array[0];
        for (int n = 1; n < array.length; n++)
            ret[n] = ret[n-1] * array[n];

        return ret;
    }

    /**
     * Calculates the reverse cumulative product of elements in an array.
     *
     * @param array an integer array.
     * @return an integer array of the same size as the input array, containing in each position the product
     *         of input elements located between this position and the end of the array, inclusive.
     */
    public static int[] reverseCumulativeProduct(int[] array) {
        int[] ret = new int[array.length];

        ret[array.length-1] = array[array.length-1];
        for (int n = array.length-2; n >= 0; n--)
            ret[n] = ret[n+1] * array[n];

        return ret;
    }

    /**
     * Generates a pair of random integers, without replacement.
     *
     * The integers are distributed uniformly.
     *
     * @param rng the random number generator used to draw numbers
     * @param maxValue an integer such that he support of the probability distribution to [0, maxValue[
     * @return an int array of length 2 with the random pair
     */
    //
    public static int[] randomPair(Random rng, int maxValue) {
        int[] pair = new int[2];
        int tmp;

        // Generate a pair
        pair[0] = rng.nextInt(maxValue);
        tmp = rng.nextInt(maxValue-1);
        pair[1] = (tmp < pair[0]) ? tmp : tmp + 1;

        return pair;
    }

}

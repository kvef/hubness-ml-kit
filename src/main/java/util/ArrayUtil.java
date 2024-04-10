
/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package util;

import data.representation.util.DataMineConstants;
import java.util.ArrayList;

/**
 * A utility class for working with arrays of measurements.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ArrayUtil {

    /**
     * Performs z-standardization of a float array.
     *
     * @param arr An array of float values.
     */
    public static void zStandardize(float[] arr) {
        if (arr == null || arr.length == 0) {
            return;
        }
        float mean = findMean(arr);
        float stDev = findStdev(arr, mean);
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (arr[i] - mean) / stDev;
        }
    }

    /**
     * Performs z-standardization of a double array.
     *
     * @param arr An array of double values.
     */
    public static void zStandardize(double[] arr) {
        if (arr == null || arr.length == 0) {
            return;
        }
        double mean = findMean(arr);
        double stDev = findStdev(arr, mean);
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (arr[i] - mean) / stDev;
        }
    }

    /**
     * Finds a mean of a float array.
     *
     * @param arr An array of float values.
     * @return Mean value.
     */
    public static float findMean(float[] arr) {
        if (arr == null || arr.length == 0) {
            return 0;
        }
        float mean;
        double totalSum = 0;
        for (int i = 0; i < arr.length; i++) {
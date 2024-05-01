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

/**
 * Calculates the non-homogeneity of an array of values.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NonHomogenityCalculator {

    /**
     * Calculates the non-homogeneity of the array.
     *
     * @param arr An array of float values.
     * @return Non-homogeneity score.
     */
    public static float calculateNonHomogeneity(float[] arr) {
        if (arr == null) {
            return Float.NaN;
        }
        int length = arr.length;
        float denominator = (float) Math.sqrt(
                (Math.pow((1 / ((float) length)), 2) * (length - 1)
                + Math.pow(((length - 1) / ((float) length)), 2))
                / ((float) length));
        float sum = 0;
        for (int i = 0; i < length; i++) {
            sum += (float) Math.pow(arr[i] - (1 / ((float) length)), 2);
        }
        sum = (float) Math.sqrt(sum / (float) length);
        float result = sum / denominator;
        return result;
    }

    /**
     * Calculates the non-homogeneity of the array.
     *
     * @param arr An array of do
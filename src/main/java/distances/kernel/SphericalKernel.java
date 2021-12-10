
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
package distances.kernel;

import data.representation.util.DataMineConstants;
import java.util.HashMap;
import java.util.Set;

/**
 * The spherical kernel is similar to the circular kernel, but is positive
 * definite in R3.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SphericalKernel extends Kernel {

    private float sigma = 1f; // Kernel width should be carefully set.

    public SphericalKernel() {
    }

    /**
     * @param sigma Kernel width.
     */
    public SphericalKernel(float sigma) {
        this.sigma = sigma;
    }

    /**
     * @param x Feature value array.
     * @param y Feature value array.
     * @return
     */
    @Override
    public float dot(float[] x, float[] y) {
        if ((x == null && y != null) || (x != null && y == null)) {
            return Float.MAX_VALUE;
        }
        if ((x == null && y == null)) {
            return 0;
        }
        if (x.length != y.length) {
            return Float.MAX_VALUE;
        }
        double result = 0;
        for (int i = 0; i < x.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(x[i])
                    || !DataMineConstants.isAcceptableFloat(y[i])) {
                continue;
            }
            result += (x[i] - y[i]) * (x[i] - y[i]);
        }
        result = Math.sqrt(result);
        if (result < sigma) {
            double sigQuot = result / sigma;
            result = 1 - (1.5 * sigQuot) + (0.5 * sigQuot * sigQuot * sigQuot);
            return (float) result;
        } else {
            return 0;
        }
    }

    /**
     * @param x Feature value sparse vector.
     * @param y Feature value sparse vector.
     * @return
     */
    public float dot(HashMap<Integer, Float> x, HashMap<Integer, Float> y) {
        if ((x == null || x.isEmpty())
                && (y == null || y.isEmpty())) {
            return 0;
        } else if ((x == null || x.isEmpty())
                && (y != null && !y.isEmpty())) {
            return Float.MAX_VALUE;
        } else if ((y == null || y.isEmpty())
                && (x != null && !x.isEmpty())) {
            return Float.MAX_VALUE;
        } else {
            Set<Integer> keysX = x.keySet();
            Set<Integer> keysY = y.keySet();
            double result = 0;
            for (int index : keysX) {
                if (y.containsKey(index)) {
                    if (DataMineConstants.isAcceptableFloat(x.get(index))
                            && DataMineConstants.isAcceptableFloat(
                            y.get(index))) {
                        result += (x.get(index) - y.get(index))
                                * (x.get(index) - y.get(index));
                    }
                } else {
                    if (DataMineConstants.isAcceptableFloat(
                            x.get(index))) {
                        result += x.get(index) * x.get(index);
                    }
                }
            }
            for (int index : keysY) {
                if (!x.containsKey(index)
                        && DataMineConstants.isAcceptableFloat(
                        y.get(index))) {
                    result += y.get(index) * y.get(index);
                }
            }
            result = Math.sqrt(result);
            if (result < sigma) {
                double sigQuot = result / sigma;
                result = 1 - (1.5 * sigQuot)
                        + (0.5 * sigQuot * sigQuot * sigQuot);
                return (float) result;
            } else {
                return 0;
            }
        }
    }
}
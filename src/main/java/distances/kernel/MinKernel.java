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
 * The Histogram Intersection Kernel is also known as the Min Kernel and has
 * been proven useful in image classification.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MinKernel extends Kernel {

    public MinKernel() {
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
            result += Math.min(x[i], y[i]);
        }
        return (float) result;
    }

    /**
     * @param x Feature value sparse vector.
     * @param y Feature value sparse vector.
     * @return
     */
    @Ov
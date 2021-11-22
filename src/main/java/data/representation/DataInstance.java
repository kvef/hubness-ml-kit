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
package data.representation;

import data.representation.util.DataMineConstants;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * The basic data class for representing data instances. The values are public
 * in the current implementation, in order to facilitate fast access and faster
 * development. The assumption is that the data is not modified by the learning
 * algorithms and that it is only written to during loading and preprocessing.
 * This is a reasonable assumption in learning evaluation systems. Therefore,
 * some care should be taken when working in a multi-threaded setting. In case
 * of applications where continuous data integrity can not be guaranteed, one
 * should wrap the data in a different container that would hide its fields.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DataInstance implements Serializable {

    private static final long serialVersionUID = 1L;
    // The DataSet that defines the feature types for this data instance.
    private DataSet dataContext = null;
    // The label of the data instance.
    private int category = 0;
    // Support for fuzzy labels is slowly being added throughout the code. Most
    // methods work with crisp labels, as is customary.
    public float[] fuzzyLabels = null;
    // Feature values.
    public int[] iAttr = null;
    public float[] fAttr = null;
    public String[] sAttr = null;
    // Identifier, which can be composed of multiple values and is thus also
    // represented as a data instance.
    private DataInstance identifier = null;

    /**
     * Noise is marked by having -1 label.
     *
     * @return True if not noise, false if noise.
     */
    public boolean notNoise() {
        return (category != -1);
    }

    /**
     * Noise is marked by having -1 label.
     *
     * @return True if noise, false if not noise.
     */
    public boolean isNoise() {
        return (category == -1);
    }

    /**
     * Sets the category of this instance to -1, which marks it as noise.
     */
    public void markAsNoise() {
        category = -1;
    }

    /**
     * In some I/O operations, an unchecked error might cause some instances to
     * be empty in a sense that they have all zero values. Not in this library,
     * but there were some cases with imported data. This method checks whether
     * an instance has all zero float values and is called in procedures that
     * check for data abnormalities in certain contexts. Of course, in a
     * different context, having all zero values might still yield a meaningful
     * representation.
     *
     * @return True if all float feature values are zero, false otherwise.
     */
    public boolean isZeroFloatVector() {
        if (hasFloatAtt()) {
            for (int i = 0; i < fAttr.length; i++) {
                if (fAttr[i] > 0) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    /**
     * @return True if the instance has float features, false otherwise.
     */
    public boolean hasFloatAtt() {
        if (fAttr == null || fAttr.length == 0) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * @return True if the instance has integer features, false otherwise.
     */
    public boolean hasIntAtt() {
        if (iAttr == null || iAttr.length == 0) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * @return True if the instance has nominal features, false otherwise.
     */
    public boolean hasNomAtt() {
        if (sAttr == null || sAttr.length == 0) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * @return The number of float features of this instance.
     */
    public int getNumFAtt() {
        if (fAttr == null) {
            return 0;
        } else {
            return fAttr.length;
        }
    }

    /**
     * @return The number of integer features of this instance.
     */
    public int getNumIAtt() {
        if (iAttr == null) {
            return 0;
        } else {
            return iAttr.length;
        }
    }

    /**
     * @return The number of nominal fea
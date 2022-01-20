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
package feature.evaluation;

import data.representation.discrete.DiscretizedDataSet;
import data.representation.util.DataMineConstants;
import java.util.ArrayList;

/**
 * This class implements the basic functionality for discrete attribute
 * evaluation and is extended by concrete approach implementations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class DiscreteAttributeEvaluator {

    private DiscretizedDataSet discDSet = null;
    private float[] intEvaluations;
    private float[] floatEvaluations;
    private float[] nominalEvaluations;

    /**
     * @param discDSet Discretized dataset that is being analyzed.
     */
    public void setDiscretizedDataSet(DiscretizedDataSet discDSet) {
        this.discDSet = discDSet;
    }

    /**
     * @return Discretized dataset that is being analyzed.
     */
    public DiscretizedDataSet getDiscretizedDataSet() {
        return discDSet;
    }

    /**
     * Evaluate the specified feature.
     *
     * @param attType Integer that is the feature type, as in DataMineConstants.
     * @param attIndex Index of the feature in its feature group.
     * @return Float value that is the evaluation.
     */
    public abstract float evaluate(int attType, int attIndex);

    /**
     * Evaluate the specified feature.
     *
     * @param subset Indexes of the instance subset to evaluate the feature on.
     * @param attType Integer that is the feature type, as in DataMineConstants.
     * @param attIndex Index of the feature in its feature group.
     * @return Float value that is the evaluation.
     */
    public abstract float evaluateOnSubset(ArrayList<Integer> subset,
            int attType, int attIndex);

    public abstract float evaluateSplit(ArrayList<Integer>[] split);

    public abstract float evaluateSplitOnSubset(ArrayList<Integer> subset,
            ArrayList<Integer>[] split);

    /**
     * Get the type and index of the feature with the lowest evaluation.
     *
     * @return Integer array of two elements, the first one being the type an
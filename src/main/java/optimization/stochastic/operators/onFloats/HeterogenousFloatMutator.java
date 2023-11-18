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
package optimization.stochastic.operators.onFloats;

import data.representation.DataInstance;
import data.representation.util.DataMineConstants;
import java.util.Random;
import optimization.stochastic.operators.HeterogenousMutationInterface;

/**
 * Heterogenous float mutator, mutating float features in such a way that
 * different features have different mutations rates.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HeterogenousFloatMutator implements HeterogenousMutationInterface {

    private float pMutation = 1;
    private float[] stDev;
    private float[] lowerBounds;
    private float[] upperBounds;
    private int beginIndex = -1;
    private int endIndex = -1;

    /**
     * @param stDev Array of mutation standard deviations.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     */
    public HeterogenousFloatMutator(
            float[] stDev,
            float[] lowerBounds,
            float[] upperBounds) {
        this.stDev = stDev;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
    }

    /**
     * @param stDev Array of mutation standard deviations.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param beginIndex The index of the first feature to mutate.
     * @param endIndex The index of the last feature to mutate.
     */
    public HeterogenousFloatMutator(
            float[] stDev,
            float[] lowerBounds,
            float[] upperBounds,
            int beginIndex,
            int endIndex) {
        this.stDev = stDev;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.beginIndex = beginIndex;
        this.endIndex = endIndex;
    }

    /**
     * @param stDev Array of mutation standard deviations.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param pMutation Probability of whether to mutate a feature.
     */
    public HeterogenousFloatMutator(
            float[] stDev,
            float[] lowerBounds,
            float[] upperBounds,
            float pMutation) {
        this.stDev = stDev;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.pMutation = pMutation;
    }

    /**
     * @param stDev Array of mutation standard deviations.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param beginIndex The index of the first feature to mutate.
     * @param endIndex The index of the last feature to mutate.
     * @param pMutation Probability of whether to mutate a feature.
     */
    public HeterogenousFloatMutator(
            float[] stDev,
            float[] lowerBounds,
            float[] upperBound
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
import optimization.stochastic.operators.TwoDevsMutationInterface;

/**
 * A class that implements the homogenous two deviations float mutator. It is
 * homogenous in a sense that each feature has the same mutation probability. A
 * separate probability constant controls the chances of making small random
 * mutations vs large random mutations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HomogenousTwoDevsFloatMutator implements TwoDevsMutationInterface {

    private float pMutation = 1;
    private float stDevSmall;
    private float stDevBig;
    private float pSmall;
    private float[] lowerBounds;
    private float[] upperBounds;
    private int beginIndex = -1;
    private int endIndex = -1;

    /**
     * @param stDevSmall Standard deviation of small mutations.
     * @param stDevBig Standard deviation of large mutations.
     * @param pSmall Probability of making small mutations.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     */
    public HomogenousTwoDevsFloatMutator(
            float stDevSmall,
            float stDevBig,
            float pSmall,
            float[] lowerBounds,
            float[] upperBounds) {
        this.stDevSmall = stDevSmall;
        this.stDevBig = stDevBig;
        this.pSmall = pSmall;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
    }

    /**
     * @param stDevSmall Standard deviation of small mutations.
     * @param stDevBig Standard deviation of large mutations.
     * @param pSmall Probability of making small mutations.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param beginIndex The index of the first feature to mutate.
     * @param endIndex The index of the last feature to mutate.
     */
    public HomogenousTwoDevsFloatMutator(
            float stDevSmall,
            float stDevBig,
            float pSmall,
            float[] lowerBounds,
            float[] upperBounds,
            int beginIndex,
            int endIndex) {
        this.stDevSmall = stDevSmall;
        this.stDevBig = stDevBig;
        this.pSmall = pSmall;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.beginIndex = beginIndex;
        this.endIndex = endIndex;
    }

    /**
     * @param stDevSmall Standard deviation of small mutations.
     * @param stDevBig Standard deviation of large mutations.
     * @param pSmall Probability of making small mutations.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param pMutation Probability of whether to mutate a feature.
     */
    public HomogenousTwoDevsFloatMutator(
            float stDevSmall,
            float stDevBig,
            float pSmall,
            float[] lowerBounds,
            float[] upperBounds,
            float pMutation) {
        this.stDevSmall = stDevSmall;
        this.stDevBig = stDevBig;
        this.pSmall = pSmall;
        this.lowerBounds = lowerBounds;
        this.upperBounds = upperBounds;
        this.pMutation = pMutation;
    }

    /**
     * @param stDevSmall Standard deviation of small mutations.
     * @param stDevBig Standard deviation of large mutations.
     * @param pSmall Probability of making small mutations.
     * @param lowerBounds Lower value bounds.
     * @param upperBounds Upper value bounds.
     * @param beginIndex The index of the first feature to mutate.
     * @param endIndex The index of the last feature to mutate.
     * @param pMutation Probability of whether to mutate a feature.
     */
    public HomogenousTwoDevsFloatMutator(
            float stDevSmall,
            float stDevBig,
            float pSmall,
            float[] lowerBounds,
            float[] upperBounds,
            int beginIndex,
            int endIndex,
            float pMutation) {
        this.stDevSmall = stDevSmall;
        this.stDevBig = stDevBig;
        this.pSmall = pSmall;
        this.lowerBounds = lowerBounds;
        thi
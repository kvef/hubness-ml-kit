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
package preprocessing.instance_selection;

import algref.Citable;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Collections;

/**
 * IMPORTANT: InstanceSelector should not change the ordering of the elements.
 * Use the sortSelectedIndexes method at the end of the reduction. This class
 * deals with instance selection, a case where no new data representatives are
 * generated, but already existing data points are selected as prototypes
 * instead.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class InstanceSelector implements Citable {

    private int k; // Neighborhood datasize for hubness calculations.
    private CombinedMetric cmet = CombinedMetric.FLOAT_MANHATTAN;
    private DataSet originalDSet;
    private ArrayList<Integer> prototypeIndexes;
    private int numClasses;
    private int[] protoHubness;
    private int[] protoGoodHubness;
    private int[] protoBadHubness;
    // First index in the array is the class, second is the element.
    private int[][] protoClassHubness;
    private int[][] protoNeighborSets;
    
    /**
     * This method calculates the unbiased class to class hubness matrix.
     * Normalization is done by the class hubness counts.
     * @return float[][] matrix representing class-to-class hubness.
     */
    public float[][] calculateClassToClassPriorsFuzzy() {
        if (protoNeighborSets == null) {
            return null;
        }
        float[][] classToClassPriors = new float[numClasses][numClasses];
        float[] classHubnessSums = new float[numClasses];
        for (int i = 0; i < originalDSet.size(); i++) {
            int currClass = originalDSet.getLabelOf(i);
            for (int kIndex = 0; kIndex < k; kIndex++) {
                int neighborClass = originalDSet.getLabelOf(
                        protoNeighborSets[i][kIndex]);
                classToClassPriors[neighborClass][currClass]++;
                classHubnessSums[neighborClass]++;
            }
        }
        float laplaceEstimator = 0.001f;
        float laplaceTotal = numClasses * laplaceEstimator;
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                classToClassPriors[i][j] += laplaceEstimator;
                classToClassPriors[i][j] /=
                        (classHubnessSums[i] + laplaceTotal);
            }
        }
        return classToClassPriors;
    }
    
    /**
     * This method calculates the unbiased class to class hubness matrix.
     * Normalization is done by the total class counts.
     * @return float[][] matrix representing class-to-class hubness.
     */
    public float[][] calculateClassToClassPriorsBayesian() {
        if (protoNeighborSets == null) {
            return null;
        }
        float[][] classToClassPriors = new float[numClasses][numClasses];
        float[] classFreqs = originalDSet.getClassFrequenciesAsFloatArray();
        for (int i = 0; i < originalDSet.size(); i++) {
            int currClass = originalDSet.getLabelOf(i);
            for (int kIndex = 0; kIndex < k; kIndex++) {
                int neighborClass = originalDSet.getLabelOf(
                        protoNeighborSets[i][kIndex]);
                classToClassPriors[neighborClass][currClass]++;
            }
        }
        float laplaceEstimator = 0.001f;
        float laplaceTotal = numClasses * laplaceEstimator;
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                classToClassPriors[i][j] += laplaceEstimator;
                classToClassPriors[i][j] /=
                        (classFreqs[i] + laplaceTotal);
            }
        }
        return classToClassPriors;
    }
    
    /**
     * @return Integer value that is the original data size. 
     */
    public int getOriginalDataSize() {
        if (originalDSet == null) {
            return 0;
        } else {
            return originalDSet.size();
        }
    }

    /**
     * @return Neighborhood size used for hubness calculations.
     */
    public int getNeighborhoodSize() {
        return k;
    }

    /**
     * @param k Integer that represents neighborhood size to be used for hubness
     * calculations.
     */
    public void setNeighborhoodSize(int k) {
        this.k = k;
    }

    /**
     * @param cmet CombinedMetric object.
     */
    public void setCombinedMetric(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * @return CombinedMetric object.
     */
    public CombinedMetric getCombinedMetric() {
        return cmet;
    }

    /**
     * Sorts selected indexes.
     */
    public void sortSelectedIndexes() {
        Collections.sort(prototypeIndexes);
    }

    /**
     * @return The original, non-reduced DataSet.
     */
    public DataSet getOriginalDataSet() {
        return originalDSet;
    }

    /**
     * @param originalDSet The original, non-reduced DataSet.
     */
    public void setOriginalDataSet(DataSet originalDSet) {
        this.originalDSet = originalDSet;
        if (originalDSet != null) {
            numClasses = originalDSet.countCategories();
        }
    }

    /**
     * @return Integer array representing the neighbor occurrence frequencies of
     * the selected prototypes.
     */
    public int[] getPrototypeHubness() {
        return protoHubness;
    }

    /**
     * @return Integer array representing the good neighbor occurrence
     * frequencies of the selected prototypes, where label mismatches do not
     * occur.
     */
    public int[] getPrototypeGoodHubness() {
        return protoGoodHubness;
    }

    /**
     * @return Integer array representing the bad neighbor occurrence
     * frequencies of the selected prototypes, where label mismatches occur.
     */
    public int[] getPrototypeBadHubness() {
        return protoBadHubness;
    }

    /**
     * @return An integer 2d array, containing the class-conditional neighbor
     * occurrence frequencies of the selected prototypes. The first index in the
     * array corresponds to the class and the second one to the prototype.
     */
    public int[][] getProtoClassHubness() {
        return protoClassHubness;
    }

    /**
     * @return Integer 2d array representing the kNN sets of the selected
     * prototypes.
     */
    public int[][] getProtoNeighborSets() {
        return protoNeighborSets;
    }

    /**
     * @param protoHubness Integer array representing the neighbor occurrence
     * frequencies of the selected prototypes.
     */
    public void setPrototypeHubness(int[] protoHubness) {
        this.protoHubness = protoHubness;
    }

    /**
     * @param protoGoodHubness Integer array representing the good neighbor
     * occurrence frequencies of the selected prototypes, where label mismatches
     * do not occur.
     */
    public void setPrototypeGoodHubness(int[] protoGoodHubness) {
        this.protoGoodHubness = protoGoodHubness;
    }

    /**
     * @param protoBadHubness Integer array representing the bad neighbor
     * occurrence frequencies of the selected prototypes, where label mismatches
     * occur.
     */
    public void setPrototypeBadHubness(int[] protoBadHubness) {
        this.protoBadHubness = protoBadHubness;
    }

    /**
     * @param protoClassHubness An integer 2d array, containing the
     * class-conditional neighbor occurrence frequencies of the selected
     * prototypes. The first index in the array corresponds to the class and the
     * second one to the prototype.
     */
    public void setProtoClassHubness(int[][] protoClassHubness) {
        this.protoClassHubness = protoClassHubness;
    }

    /**
     * @param protoNeighborSets Integer 2d array
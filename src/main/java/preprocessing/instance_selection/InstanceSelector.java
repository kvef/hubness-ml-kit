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
     * @param protoNeighborSets Integer 2d array representing the kNN sets of
     * the selected prototypes.
     */
    public void setProtoNeighborSets(int[][] protoNeighborSets) {
        this.protoNeighborSets = protoNeighborSets;
    }

    /**
     * @return The number of classes in the data.
     */
    public int getNumClasses() {
        return numClasses;
    }

    /**
     * @param numClasses The number of classes in the data.
     */
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    /**
     * @param prototypeIndexes Indexes of the selected prototypes in the
     * original data set.
     */
    public void setPrototypeIndexes(ArrayList<Integer> prototypeIndexes) {
        this.prototypeIndexes = prototypeIndexes;
    }

    /**
     * @return Indexes of the selected prototypes in the original data set.
     */
    public ArrayList<Integer> getPrototypeIndexes() {
        return prototypeIndexes;
    }

    /**
     * Performs data reduction by automatically determining the proper reduction
     * rate.
     *
     * @throws Exception
     */
    public abstract void reduceDataSet() throws Exception;

    /**
     * Performs data reduction to a predetermined number of prototypes.
     *
     * @param numPrototypes Number of prototypes to reduce the data to.
     * @throws Exception
     */
    public abstract void reduceDataSet(int numPrototypes) throws Exception;

    /**
     * Reduce the data set according to the specified reduction/retainment rate.
     *
     * @param percRetained Percentage of prototypes to be retained.
     * @throws Exception
     */
    public void reduceDataSet(float percRetained) throws Exception {
        float datasize = originalDSet.size();
        int numPrototypes = (int) (datasize * percRetained);
        reduceDataSet(numPrototypes);
    }

    /**
     * @param copyInstances Boolean flag indicating whether to make object
     * copies of the selected prototypes when generating the reduced DataSet or
     * not, in which case the original objects are used instead.
     * @return DataSet that contains only the selected prototypes.
     * @throws Exception
     */
    public DataSet getReducedDataSet(boolean copyInstances) throws Exception {
        if (originalDSet == null) {
            return null;
        }
        DataSet reducedDS = originalDSet.cloneDefinition();
        if (prototypeIndexes == null || prototypeIndexes.isEmpty()) {
            return reducedDS;
        }
        reducedDS.data = new ArrayList<>(prototypeIndexes.size());
        DataInstance instance;
        for (int i : prototypeIndexes) {
            if (copyInstances) {
                instance = originalDSet.getInstance(i).copy();
                instance.embedInDataset(reducedDS);
                reducedDS.addDataInstance(instance);
            } else {
                reducedDS.addDataInstance(originalDSet.getInstance(i));
            }
        }
        return reducedDS;
    }

    /**
     * @return A copy of the current InstanceSelector object.
     */
    public abstract InstanceSelector copy();

    /**
     * Calculates the neighbor occurrence profiles of the selected prototypes.
     * This is meant to be overridden in those subclasses which use NSF-s
     * (already have a kNN graph) - as this is slower, it assumes no previous
     * neighbor information.
     *
     * @param k Neighborhood size to be used in hubness calculations.
     * @throws Exception
     */
    public void calculatePrototypeHubness(int k) throws Exception {
        if (k <= 0) {
            return;
        }
        this.k = k;
        int[][] kneighbors = new int[originalDSet.size()][k];
        // Make a data subset.
        DataSet 
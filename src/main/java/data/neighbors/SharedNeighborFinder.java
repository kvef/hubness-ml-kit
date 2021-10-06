
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
package data.neighbors;

import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import util.ArrayUtil;
import util.BasicMathUtil;

/**
 * This class calculates the kNN set intersections between different points in
 * order to later calculate shared-neighbor similarity/dissimilarity measures.
 * These secondary measures are often better suited for high-dimensional data
 * than the primary metrics.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SharedNeighborFinder implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private NeighborSetFinder nsf = null;
    private DataSet dset = null;
    private CombinedMetric cmet = null;
    // This neighborhood size is used for finding shared neighbors.
    private int k = 50;
    // This neigborhood size is used for calculating instance weights in
    // certain weighting schemes.
    private int kClassification = 5;
    // Hash the neighbors of each point for faster calculations of the
    // intersections.
    private HashMap<Integer, Integer>[] neighborHash;
    private int numClasses;
    // Instance weights to use in distance calculations.
    float[] instanceWeights;
    // Shared neighbor lists.
    private ArrayList<Integer>[][] sharedNeighbors = null;
    // Shared neighbor counts.
    private float[][] sharedNeighborCount = null;
    public static final int DEFAULT_NUM_THREADS = 8;

    /**
     * Initializes the neighbor hashes.
     */
    private void initializeHashes() {
        if (dset == null || nsf == null) {
            return;
        }
        neighborHash = new HashMap[dset.size()];
        int[][] kneighbors = nsf.getKNeighbors();
        for (int i = 0; i < dset.size(); i++) {
            neighborHash[i] = new HashMap<>(2 * k);
            for (int j = 0; j < k; j++) {
                neighborHash[i].put(kneighbors[i][j], j);
            }
        }
    }

    /**
     * Checks if neighbor hashes have been initialized.
     *
     * @return True if the hashes are initialized, false otherwise.
     */
    private boolean hashInitialized() {
        if (neighborHash == null || neighborHash.length == 0) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * @param nsf NeighborSetFinder object.
     */
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    /**
     * @return NeighborSetFinder object.
     */
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    /**
     * Calculates the instance weights for the class-imbalanced case.
     */
    public void obtainWeightsForClassImbalancedData() {
        int size = dset.size();
        float[] weightArray = new float[size];
        int numCat = dset.countCategories();
        int[] classCounts = dset.getClassFrequencies();
        float maxEntropy = (float) BasicMathUtil.log2((float) numCat);
        float[][] chubness = nsf.getClassDataNeighborRelationNonNormalized(
                k, numCat, false);
        float[][] classToClass = nsf.getGlobalClassToClassNonNormalized(
                kClassification, numCat);
        float[] classRelevance = new float[numCat];
        for (int c = 0; c < numCat; c++) {
            classRelevance[c] = (float) (1 - ((classToClass[c][c] + 0.00001)
                    / (kClassification * classCounts[c] + 0.00001)));
        }
        if (nsf.getReverseNeighborEntropies() == null) {
            nsf.calculateReverseNeighborEntropiesWeighted(numCat,
                    classRelevance);
        }
        int[] neighbOccFreqs = nsf.getNeighborFrequencies();
        float maxWeight = 1;
        for (int i = 0; i < size; i++) {
            weightArray[i] = (float) BasicMathUtil.log2(((float) size)
                    / ((float) nsf.getNeighborFrequencies()[i] + 1));
            maxWeight = Math.max(Math.abs(weightArray[i]), maxWeight);
        }
        for (int i = 0; i < size; i++) {
            weightArray[i] /= maxWeight;
        }
        // This was the I_n / max I_n part of the weight.              
        maxWeight = 0;
        float min = Float.MAX_VALUE;
        // Difference between good and bad hubness.
        float indicator;
        // Good and bad hubness sums.
        double ghsum;
        double bhsum;
        float[] weightArraySecond = new float[size];
        float[] ghsumAll = new float[size];
        float[] bhsumAll = new float[size];

        for (int i = 0; i < size; i++) {
            if (neighbOccFreqs[i] < 1) {
                weightArraySecond[i] = 1;
                continue;
            }
            ghsum = 0;
            bhsum = 0;
            for (int c = 0; c < numCat; c++) {
                ghsum += (chubness[c][i] * (chubness[c][i] - 1) / 2)
                        * (1 - ((classToClass[c][c] + 0.00001)
                        / (kClassification * classCounts[c] + 0.00001)));
                for (int c1 = 0; c1 < numCat; c1++) {
                    if (c != c1) {
                        bhsum += (chubness[c][i] * chubness[c1][i] * 0.5
                                * (((classToClass[c1][c] + 0.00001)
                                / (kClassification * classCounts[c] + 0.00001))
                                + ((classToClass[c][c1] + 0.00001)
                                / (kClassification * classCounts[c1]
                                + 0.00001))));
                    }
                }
            }
            ghsumAll[i] = (float) ghsum;
            bhsumAll[i] = (float) bhsum;
            indicator = (float) (ghsum - bhsum);
            weightArraySecond[i] = indicator;
            maxWeight = Math.max(weightArraySecond[i], maxWeight);
            min = Math.min(weightArraySecond[i], min);
        }
        ArrayUtil.zStandardize(weightArraySecond);
        ArrayUtil.zStandardize(ghsumAll);
        ArrayUtil.zStandardize(bhsumAll);
        for (int i = 0; i < size; i++) {
            weightArraySecond[i] = 1f
                    / (1f + (float) Math.exp(-weightArraySecond[i]));
        }
        for (int i = 0; i < size; i++) {
            ghsumAll[i] = 1f / (1f + (float) Math.exp(-ghsumAll[i]));
        }
        for (int i = 0; i < size; i++) {
            bhsumAll[i] = 1f / (1f + (float) Math.exp(bhsumAll[i]));
        }

        maxWeight = 1;
        float[] weightArrayProfile = new float[size];
        for (int i = 0; i < size; i++) {
            weightArrayProfile[i] = (maxEntropy
                    - nsf.getReverseNeighborEntropies()[i]);
            maxWeight = Math.max(Math.abs(weightArray[i]), maxWeight);
        }
        for (int i = 0; i < size; i++) {
            weightArrayProfile[i] /= maxWeight;
        }

        for (int i = 0; i < size; i++) {
            weightArray[i] = weightArray[i] * weightArrayProfile[i]
                    * weightArraySecond[i];
        }
        instanceWeights = weightArray;
    }

    /**
     * @param cmet CombinedMetric object that is the primary distance measure.
     */
    public void setPrimaryMetricsCalculator(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * @param numClasses Integer that is the number of classes.
     */
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    /**
     *
     */
    public SharedNeighborFinder() {
    }

    /**
     *
     * @param dset DataSet that is being analyzed.
     * @param k Neighborhood size for shared-neighbor calculation.
     * @param cmet CombinedMetric object for primary distances.
     * @param kClassification Neighborhood size used for classification.
     */
    public SharedNeighborFinder(DataSet dset,
            int k, CombinedMetric cmet, int kClassification) {
        this.dset = dset;
        this.k = k;
        this.cmet = cmet;
        this.kClassification = kClassification;
    }

    /**
     *
     * @param dset DataSet that is being analyzed.
     * @param k Neighborhood size for shared-neighbor calculation.
     * @param cmet CombinedMetric object for primary distances.
     */
    public SharedNeighborFinder(DataSet dset, int k, CombinedMetric cmet) {
        this.dset = dset;
        this.k = k;
        this.cmet = cmet;
    }

    /**
     *
     * @param nsf NeighborSetFinder containing the kNN information.
     */
    public SharedNeighborFinder(NeighborSetFinder nsf) {
        this.nsf = nsf;
        this.k = nsf.getCurrK();
        this.dset = nsf.getDataSet();
        this.cmet = nsf.getCombinedMetric();
        initializeHashes();
    }

    /**
     *
     * @param nsf NeighborSetFinder containing the kNN information.
     * @param kClassification Neighborhood size that will be later used in
     * classification. It is sometimes used for finding instance weights. The k
     * that will be used for shared-neighbor calculations is extracted from the
     * NeighborSetFinder object instead.
     */
    public SharedNeighborFinder(NeighborSetFinder nsf, int kClassification) {
        this.nsf = nsf;
        this.k = nsf.getCurrK();
        this.dset = nsf.getDataSet();
        this.cmet = nsf.getCombinedMetric();
        this.kClassification = kClassification;
        initializeHashes();
    }

    /**
     * Sets the neighborhood size that will be used in shared-neighbor
     * calculations.
     *
     * @param k Integer that is the neighborhood size that will be used in
     * shared-neighbor calculations.
     */
    public void setSNK(int k) {
        this.k = k;
    }

    /**
     * @param kClassification Neighborhood size that will be later used in
     * classification. It is sometimes used for finding instance weights.
     */
    public void setKClassification(int kClassification) {
        this.kClassification = kClassification;
    }

    /**
     * @return A float matrix that has stores the counts of shared neighbors
     * between pairs of points. It is symmetric, so we use the upper diagonal
     * form, where each row stores only d(i, j) for i > j.
     */
    public float[][] getSharedNeighborCounts() {
        return sharedNeighborCount;
    }

    /**
     * @return CombinedMetric object for primary distance calculations.
     */
    public CombinedMetric getCombinedMetric() {
        return cmet;
    }

    /**
     * @return DataSet that is being analyzed.
     */
    public DataSet getData() {
        return dset;
    }

    /**
     * @return Integer that is the neighborhood size that will be used in
     * shared-neighbor calculations.
     */
    public int getSNK() {
        return k;
    }

    /**
     * Elements that are hubs contribute less to similarity between other points
     * when found as shared neighbors.
     */
    public void obtainWeightsFromGeneralHubness() {
        instanceWeights = nsf.getPenalizeHubnessWeightingScheme();
    }

    /**
     * Elements that occur in neighbor sets between several categories are going
     * to have their weights reduced down, thereby increasing intraclass
     * similarity and decreasing interclass similarity.
     */
    public void obtainWeightsFromBadHubness() {
        instanceWeights = nsf.getHWKNNWeightingScheme();
    }

    /**
     * Get the overall hubness information instance weighting scheme.
     *
     * @param theta The tradeoff parameter.
     */
    public void obtainWeightsFromHubnessInformation(float theta) {
        if (numClasses > 0) {
            instanceWeights = nsf.getSimhubWeightingScheme(
                    numClasses, theta);
        } else {
            instanceWeights = nsf.getSimhubWeightingScheme(theta);
        }
    }

    /**
     * Get the overall hubness information instance weighting scheme while using
     * default tradeoff.
     */
    public void obtainWeightsFromHubnessInformation() {
        if (numClasses > 0) {
            instanceWeights = nsf.getSimhubWeightingScheme(
                    numClasses, 0);
        } else {
            instanceWeights = nsf.getSimhubWeightingScheme(0);
        }
    }

    /**
     * Sets the instance weights.
     *
     * @param instanceWeights Float array of instance weights.
     */
    public void setWeights(float[] instanceWeights) {
        this.instanceWeights = instanceWeights;
    }

    /**
     * @return Float array of instance weights.
     */
    public float[] getInstanceWeights() {
        return instanceWeights;
    }

    /**
     * Remove the currently used instance weights, reset them.
     */
    public void removeWeights() {
        instanceWeights = null;
    }

    /**
     * Counts the number of shared k-nearest neighbors between the two points.
     *
     * @param instanceFirst First DataInstance.
     * @param instanceSecond Second DataInstance.
     * @param kNeighborsFirst Integer array as the kNN set of the first data
     * instance.
     * @param kNeighborsSecond Integer array as the kNN set of the second data
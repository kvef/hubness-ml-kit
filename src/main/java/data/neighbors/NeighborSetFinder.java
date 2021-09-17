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

import data.neighbors.approximate.AppKNNGraphLanczosBisection;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import util.ArrayUtil;
import util.BasicMathUtil;
import util.SOPLUtil;

/**
 * This class implements the functionality for exact kNN search and kNN graph
 * calculations, in various contexts. It also implements the functionality for
 * calculating the neighbor occurrence frequencies, good and bad occurrences,
 * reverse neighbor sets, reverse and direct neighbor set entropies and other
 * hubness-related measures. Functionally, it implements various hubness-based
 * weighting modes and the class-conditional probabilistic model estimates for
 * hubness-aware classification. It is a simple implementation in that the
 * default kNN search and graph construction methods do not rely on additional
 * spatial indexing. The reason for that, though - is that this library is meant
 * primarily for high-dimensional data analysis, where such indexes have been
 * shown to be of little use - and calculating them takes time. In case of large
 * low-to-medium dimensional datasets where spatial indexing can be very useful,
 * alternative implementations should be used. This one is meant for high-dim
 * data instead. Also, in case of large-scale datasets, approximate kNN
 * extensions are to be preferred.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NeighborSetFinder implements Serializable {

    private static final long serialVersionUID = 1L;
    // Dataset to calculate the k-nearest neighbor sets from.
    private DataSet dset = null;
    // The upper triangular distance matrix, as used throughout the library.
    private float[][] distMatrix = null;
    // CombinedMetric object for distance calculations.
    private CombinedMetric cmet = null;
    // The k-nearest neighbor sets. Each row in the table contains the indexes
    // of the k-nearest neighbors for the given point in the provided DataSet.
    private int[][] kNeighbors = null;
    // Distances to the k-nearest neighbors for each point.
    private float[][] kDistances = null;
    // The current length of the kNN sets, used during kNN calculations.
    private int[] kCurrLen = null;
    // The neighbor occurrence frequencies.
    private int[] kNeighborFrequencies = null;
    // The bad neighbor occurrence frequencies.
    private int[] kBadFrequencies = null;
    // The good neighbor occurrence frequencies.
    private int[] kGoodFrequencies = null;
    // Reverse neighbor sets.
    private ArrayList<Integer>[] reverseNeighbors = null;
    // Boolean flag indicating whether the distance matrix was provided.
    private boolean distancesCalculated = false;
    // Variance of the distance values.
    private double distVariance = 0;
    // Mean of the distance value.
    private double distMean = 0;
    // Mean of the neighbor occurrence frequency.
    private double meanOccFreq;
    // Standard deviation of the neighbor occurrence frequency.
    private double stDevOccFreq;
    // Mean of the detrimental occurrence frequency.
    private double meanOccBadness = 0;
    // Standard deviation of the detrimental occurrence frequency.
    private double stDevOccBadness = 0;
    // Mean of the beneficial neighbor occurrence frequency.
    private double meanOccGoodness = 0;
    // Standard deviation of the beneficial neighbor occurrence frequency.
    private double stDevOccGoodness = 0;
    // Mean of the difference between the good and the bad occurrence counts.
    private double meanGoodMinusBadness = 0;
    // Mean of the normalized difference between the good and bad occurrence
    // counts.
    private double meanRelativeGoodMinusBadness = 0;
    // Standard deviation of the difference between the good and the bad
    // occurrence counts.
    private double stDevGoodMinusBadness = 0;
    // Standard deviation of the normalized difference between the good and bad
    // occurrence counts.
    private double stDevRelativeGoodMinusBadness = 0;
    // Entropies of the direct kNN sets.
    private float[] kEntropies = null;
    // Entropies of the reverse kNN sets.
    private float[] kRNNEntropies = null;
    // The currently operating neighborhood size.
    private int currK;
    // Small datasets can be extended by synthetic instances from the Gaussian
    // data model.

    /**
     * The default constructor.
     */
    public NeighborSetFinder() {
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that holds the data to calculate the kNN sets
     * for.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public NeighborSetFinder(DataSet dset, CombinedMetric cmet) {
        this.dset = dset;
        this.cmet = cmet;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that holds the data to calculate the kNN sets
     * for.
     * @param distMatrix float[][] that is the upper triangular distance matrix.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public NeighborSetFinder(DataSet dset, float[][] distMatrix,
            CombinedMetric cmet) {
        this.dset = dset;
        this.distMatrix = distMatrix;
        this.cmet = cmet;
        if (distMatrix == null) {
            try {
                distMatrix = dset.calculateDistMatrix(cmet);
            } catch (Exception e) {
            }
        }
        distancesCalculated = true;
        try {
            calculateOccFreqMeanAndVariance();
        } catch (Exception e) {
            System.err.println("NSF constructor error.");
            System.err.println(e.getMessage());
        }
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that holds the data to calculate the kNN sets
     * for.
     * @param distMatrix float[][] that is the upper triangular distance matrix.
     */
    public NeighborSetFinder(DataSet dset, float[][] distMatrix) {
        this.dset = dset;
        this.distMatrix = distMatrix;
        this.cmet = CombinedMetric.FLOAT_EUCLIDEAN;
        distancesCalculated = true;
        try {
            calculateOccFreqMeanAndVariance();
        } catch (Exception e) {
            System.err.println("NSF constructor error.");
            System.err.println(e.getMessage());
        }
    }

    /**
     * @return True if the distance matrix is already available, false
     * otherwise.
     */
    public boolean distancesCalculated() {
        return distancesCalculated;
    }

    /**
     * This method persists the calculated kNN sets to a file.
     *
     * @param outFile File to save the kNN sets to.
     * @throws Exception
     */
    public void saveNeighborSets(File outFile) throws Exception {
        FileUtil.createFile(outFile);
        try (PrintWriter pw = new PrintWriter(new FileWriter(outFile));) {
            if (kNeighbors != null && kNeighbors.length > 0) {
                pw.println("size:" + kNeighbors.length);
                pw.println("k:" + kNeighbors[0].length);
                for (int i = 0; i < kNeighbors.length; i++) {
                    SOPLUtil.printArrayToStream(kNeighbors[i], pw);
                    SOPLUtil.printArrayToStream(kDistances[i], pw);
                }
            } else {
                pw.println("size:0");
                pw.println("k:0");
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    /**
     * This method loads the NeighborSetFinder object from a kNN set file.
     *
     * @param inFile File where the kNN sets were persisted.
     * @param dset DataSet object that the kNN sets point to.
     * @return NeighborSet finder that was loaded from the disk.
     * @throws Exception
     */
    public static NeighborSetFinder loadNSF(File inFile, DataSet dset)
            throws Exception {
        NeighborSetFinder loadedNSF = new NeighborSetFinder();
        loadedNSF.loadNeighborSets(inFile, dset);
        return loadedNSF;
    }

    /**
     * This method loads the kNN sets from a file.
     *
     * @param inFile File where the kNN sets were persisted.
     * @param dset DataSet object that the kNN sets point to.
     * @throws Exception
     */
    public void loadNeighborSets(File inFile, DataSet dset) throws Exception {
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(inFile)));) {
            this.dset = dset;
            String line;
            String[] lineItems;
            line = br.readLine();
            // Get data size.
            int size = Integer.parseInt((line.split(":"))[1]);
            line = br.readLine();
            // Get the neighborhood size.
            int k = Integer.parseInt((line.split(":"))[1]);
            currK = k;
            // Initialize the arrays.
            kNeighbors = new int[size][k];
            kDistances = new float[size][k];
            kCurrLen = new int[size];
            kNeighborFrequencies = new int[size];
            kBadFrequencies = new int[size];
            kGoodFrequencies = new int[size];
            kEntropies = new float[size];
            kRNNEntropies = new float[size];
            reverseNeighbors = new ArrayList[size];
            // Initialize the reverse neighbor lists.
            for (int i = 0; i < size; i++) {
                reverseNeighbors[i] = new ArrayList<>(k);
            }
            int label;
            for (int i = 0; i < size; i++) {
                label = dset.getLabelOf(i);
                // The following line holds the kNN set.
                line = br.readLine();
                // The neighbor indexes are split by empty spaces.
                lineItems = line.split(" ");
                kCurrLen[i] = Math.min(lineItems.length, k);
                for (int kInd = 0; kInd < kCurrLen[i]; kInd++) {
                    // Parse the neighbor index. 
                    kNeighbors[i][kInd] = Integer.parseInt(lineItems[kInd]);
                    // Updated the good and bad occurrence counts.
                    if (label == dset.getLabelOf(kNeighbors[i][kInd])) {
                        kGoodFrequencies[kNeighbors[i][kInd]]++;
                    } else {
                        kBadFrequencies[kNeighbors[i][kInd]]++;
                    }
                    // Update the total occurrence count.
                    kNeighborFrequencies[kNeighbors[i][kInd]]++;
                    // Update the reverse neighbor list.
                    reverseNeighbors[kNeighbors[i][kInd]].add(i);
                }
                // The following line holds the distances to the previous kNN
                // set.
                line = br.readLine();
                lineItems = line.split(" ");
                for (int kInd = 0; kInd < Math.min(kCurrLen[i],
                        lineItems.length); kInd++) {
                    kDistances[i][kInd] = Float.parseFloat(lineItems[kInd]);
                }
            }
            calculateHubnessStats(false);
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
    }

    /**
     * @param k Integer that is the queried neighborhood size.
     * @return True if the calculated kNN sets' length exceeds k, false
     * otherwise.
     */
    public boolean isCalculatedUpToK(int k) {
        return (kNeighbors != null && kDistances != null
                && kNeighbors.length >= k && kDistances.length >= k);
    }

    /**
     * This method calculates the average distance to the k-nearest neighbors
     * for each point for the specified neighborhood size.
     *
     * @param k Integer that is the neighborhood size to calculate the average
     * k-distances for.
     * @return float[] representing the average distance to the k-nearest
     * neighbors for each point in the data.
     */
    public float[] getAvgDistToNeighbors(int k) {
        if (k <= 0) {
            return null;
        }
        if (k > kDistances[0].length) {
            k = kDistances[0].length;
        }
        float[] avgKDists = new float[kDistances.length];
        for (int i = 0; i < kDistances.length; i++) {
            avgKDists[i] = 0;
            for (int kInd = 0; kInd < k; kInd++) {
                avgKDists[i] += kDistances[i][kInd];
            }
            avgKDists[i] /= k;
        }
        return avgKDists;
    }

    /**
     * @return Integer that is the currently operating neighborhood size.
     */
    public int getCurrK() {
        return currK;
    }

    /**
     * @param distMatrix float[][] representing the upper triangular distance
     * matrix, where the length of each row i is (size - i - 1) and each row
     * contains only the entries for j > i, so that distMatrix[i][j] represents
     * the distance between i and i + j + 1.
     */
    public void setDistances(float[][] distMatrix) {
        this.distMatrix = distMatrix;
        distancesCalculated = true;
        calculateOccFreqMeanAndVariance();
    }

    /**
     * @return NeighborSetFinder that is the copy of this NeighborSetFinder
     * object.
     */
    public NeighborSetFinder copy() {
        NeighborSetFinder nsfCopy = new NeighborSetFinder();
        // Shallow copies of the DataSet, the distances and the metric object.
        nsfCopy.dset = dset;
        nsfCopy.cmet = cmet;
        nsfCopy.distMatrix = distMatrix;
        // Copy the k-nearest neighbor sets.
        if (kNeighbors != null) {
            nsfCopy.kNeighbors = new int[kNeighbors.length][];
            for (int i = 0; i < kNeighbors.length; i++) {
                nsfCopy.kNeighbors[i] = Arrays.copyOf(kNeighbors[i],
                        kNeighbors[i].length);
            }
        }
        // Copy the distances to the k-nearest neighbors.
        if (kDistances != null) {
            nsfCopy.kDistances = new float[kDistances.length][];
            for (int i = 0; i < kDistances.length; i++) {
                nsfCopy.kDistances[i] = Arrays.copyOf(kDistances[i],
                        kDistances[i].length);
            }
        }
        // Copy the current k-lengths.
        if (kCurrLen != null) {
            nsfCopy.kCurrLen = Arrays.copyOf(kCurrLen, kCurrLen.length);
        }
        // Copy the occurrence frequency and the entropy arrays.
        if (kNeighborFrequencies != null) {
            nsfCopy.kNeighborFrequencies = Arrays.copyOf(kNeighborFrequencies,
                    kNeighborFrequencies.length);
        }
        if (kBadFrequencies != null) {
            nsfCopy.kBadFrequencies = Arrays.copyOf(kBadFrequencies,
                    kBadFrequencies.length);
        }
        if (kGoodFrequencies != null) {
            nsfCopy.kGoodFrequencies = Arrays.copyOf(kGoodFrequencies,
                    kGoodFrequencies.length);
        }
        if (kEntropies != null) {
            nsfCopy.kEntropies = Arrays.copyOf(kEntropies, kEntropies.length);
        }
        if (kRNNEntropies != null) {
            nsfCopy.kRNNEntropies = Arrays.copyOf(kRNNEntropies,
                    kRNNEntropies.length);
        }
        // Copy the operating neighborhood size.
        nsfCopy.currK = currK;
        // Copy all the stats and flags.
        nsfCopy.distancesCalculated = distancesCalculated;
        nsfCopy.distVariance = distVariance;
        nsfCopy.distMean = distMean;
        nsfCopy.meanOccFreq = meanOccFreq;
        nsfCopy.stDevOccFreq = stDevOccFreq;
        nsfCopy.meanOccBadness = meanOccBadness;
        nsfCopy.stDevOccBadness = stDevOccBadness;
        nsfCopy.meanOccGoodness = meanOccGoodness;
        nsfCopy.stDevOccGoodness = stDevOccGoodness;
        nsfCopy.meanGoodMinusBadness = meanGoodMinusBadness;
        nsfCopy.meanRelativeGoodMinusBadness = meanRelativeGoodMinusBadness;
        nsfCopy.stDevGoodMinusBadness = stDevGoodMinusBadness;
        nsfCopy.stDevRelativeGoodMinusBadness = stDevRelativeGoodMinusBadness;
        // Copy the reverse neighbor lists.
        if (reverseNeighbors != null) {
            nsfCopy.reverseNeighbors = new ArrayList[reverseNeighbors.length];
            for (int i = 0; i < reverseNeighbors.length; i++) {
                if (reverseNeighbors[i] != null) {
                    nsfCopy.reverseNeighbors[i] =
                            new ArrayList<>(reverseNeighbors[i].size());
                    for (int p = 0; p < reverseNeighbors[i].size(); p++) {
                        nsfCopy.reverseNeighbors[i].add(
                                reverseNeighbors[i].get(p));
                    }

                }
            }
        }
        return nsfCopy;
    }

    /**
     * @param kcurrLen int[] that are the current kNN set lengths.
     */
    public void setKCurrLen(int[] kcurrLen) {
        this.kCurrLen = kcurrLen;
    }

    /**
     * @return int[] that are the current kNN set lengths.
     */
    public int[] getKCurrLen() {
        return kCurrLen;
    }

    /**
     * Sets the kNN set to this NeighborSetFinder object.
     *
     * @param kneighbors int[][] representing the k-nearest neighbors.
     * @param kDistances float[][] representing the k-distances.
     * @param kcurrLen int[] representing the current kNN set lengths (in case
     * some of them are not yet completed)
     */
    public void setKNeighbors(int[][] kneighbors, float[][] kDistances,
            int[] kcurrLen) {
        this.kNeighbors = kneighbors;
        this.kDistances = kDistances;
        int k = kneighbors[0].length;
        this.kCurrLen = kcurrLen;
        // Set the operating neighborhood size to the length of the kNN sets.
        currK = k;
        reverseNeighbors = new ArrayList[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            reverseNeighbors[i] = new ArrayList<>(10 * k);
        }
        kNeighborFrequencies = new int[kneighbors.length];
        kBadFrequencies = new int[kneighbors.length];
        kGoodFrequencies = new int[kneighbors.length];
        for (int i = 0; i < kneighbors.length; i++) {
            for (int kInd = 0; kInd < kcurrLen[i]; kInd++) {
                reverseNeighbors[kneighbors[i][kInd]].add(i);
                kNeighborFrequencies[kneighbors[i][kInd]]++;
                if (dset.data.get(i).getCategory() != dset.data.get(
                        kneighbors[i][kInd]).getCategory()) {
                    kBadFrequencies[kneighbors[i][kInd]]++;
                } else {
                    kGoodFrequencies[kneighbors[i][kInd]]++;
                }
            }
        }
        // Calculate the neighbor occurrence frequency stats.
        meanOccFreq = 0;
        stDevOccFreq = 0;
        meanOccBadness = 0;
        stDevOccBadness = 0;
        meanOccGoodness = 0;
        stDevOccGoodness = 0;
        meanGoodMinusBadness = 0;
        stDevGoodMinusBadness = 0;
        meanRelativeGoodMinusBadness = 0;
        stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            meanOccFreq += kNeighborFrequencies[i];
            meanOccBadness += kBadFrequencies[i];
            meanOccGoodness += kGoodFrequencies[i];
            meanGoodMinusBadness += kGoodFrequencies[i] - kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                meanRelativeGoodMinusBadness += ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]);
            } else {
                meanRelativeGoodMinusBadness += 1;
            }
        }
        meanOccFreq /= (float) kNeighborFrequencies.length;
        meanOccBadness /= (float) kBadFrequencies.length;
        meanOccGoodness /= (float) kGoodFrequencies.length;
        meanGoodMinusBadness /= (float) kGoodFrequencies.length;
        meanRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stDevOccFreq += ((meanOccFreq - kNeighborFrequencies[i])
                    * (meanOccFreq - kNeighborFrequencies[i]));
            stDevOccBadness += ((meanOccBadness - kBadFrequencies[i])
                    * (meanOccBadness - kBadFrequencies[i]));
            stDevOccGoodness += ((meanOccGoodness - kGoodFrequencies[i])
                    * (meanOccGoodness - kGoodFrequencies[i]));
            stDevGoodMinusBadness += ((meanGoodMinusBadness
                    - (kGoodFrequencies[i] - kBadFrequencies[i]))
                    * (meanGoodMinusBadness - (kGoodFrequencies[i]
                    - kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                stDevRelativeGoodMinusBadness += (meanRelativeGoodMinusBadness
                        - ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (meanRelativeGoodMinusBadness - ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]));
            } else {
                stDevRelativeGoodMinusBadness +=
                        (meanRelativeGoodMinusBadness - 1)
                        * (meanRelativeGoodMinusBadness - 1);
            }
        }
        stDevOccFreq /= (float) kNeighborFrequencies.length;
        stDevOccBadness /= (float) kBadFrequencies.length;
        stDevOccGoodness /= (float) kGoodFrequencies.length;
        stDevGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevOccFreq = Math.sqrt(stDevOccFreq);
        stDevOccBadness = Math.sqrt(stDevOccBadness);
        stDevOccGoodness = Math.sqrt(stDevOccGoodness);
        stDevGoodMinusBadness = Math.sqrt(stDevGoodMinusBadness);
        stDevRelativeGoodMinusBadness =
                Math.sqrt(stDevRelativeGoodMinusBadness);
    }

    /**
     * Sets the kNN set to this NeighborSetFinder object.
     *
     * @param kneighbors int[][] representing the k-nearest neighbors.
     * @param kDistances float[][] representing the k-distances.
     */
    public void setKNeighbors(int[][] kneighbors, float[][] kDistances) {
        this.kNeighbors = kneighbors;
        this.kDistances = kDistances;
        int k = kneighbors[0].length;
        kCurrLen = new int[kneighbors.length];
        // The kNN sets are completed.
        Arrays.fill(kCurrLen, k);
        // Set the operating neighborhood size.
        currK = k;
        // Initialize the reverse neighbor lists.
        reverseNeighbors = new ArrayList[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            reverseNeighbors[i] = new ArrayList<>(10 * k);
        }
        kNeighborFrequencies = new int[kneighbors.length];
        kBadFrequencies = new int[kneighbors.length];
        kGoodFrequencies = new int[kneighbors.length];
        // Fill in the reverse neighbor lists and count the occurence
        // frequencies.
        for (int i = 0; i < kneighbors.length; i++) {
            for (int kInd = 0; kInd < k; kInd++) {
                reverseNeighbors[kneighbors[i][kInd]].add(i);
                kNeighborFrequencies[kneighbors[i][kInd]]++;
                if (dset.data.get(i).getCategory() != dset.data.get(
                        kneighbors[i][kInd]).getCategory()) {
                    kBadFrequencies[kneighbors[i][kInd]]++;
                } else {
                    kGoodFrequencies[kneighbors[i][kInd]]++;
                }
            }
        }
        // Calculate the occurrence frequency stats.
        meanOccBadness = 0;
        stDevOccBadness = 0;
        meanOccGoodness = 0;
        stDevOccGoodness = 0;
        meanGoodMinusBadness = 0;
        stDevGoodMinusBadness = 0;
        meanRelativeGoodMinusBadness = 0;
        stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            meanOccBadness += kBadFrequencies[i];
            meanOccGoodness += kGoodFrequencies[i];
            meanGoodMinusBadness += kGoodFrequencies[i] - kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                meanRelativeGoodMinusBadness +=
                        ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]);
            } else {
                meanRelativeGoodMinusBadness += 1;
            }
        }
        meanOccBadness /= (float) kBadFrequencies.length;
        meanOccGoodness /= (float) kGoodFrequencies.length;
        meanGoodMinusBadness /= (float) kGoodFrequencies.length;
        meanRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        for (int i = 0; i < kBadFrequencies.length; i++) {
            stDevOccBadness += ((meanOccBadness - kBadFrequencies[i])
                    * (meanOccBadness - kBadFrequencies[i]));
            stDevOccGoodness += ((meanOccGoodness - kGoodFrequencies[i])
                    * (meanOccGoodness - kGoodFrequencies[i]));
            stDevGoodMinusBadness += ((meanGoodMinusBadness
                    - (kGoodFrequencies[i] - kBadFrequencies[i]))
                    * (meanGoodMinusBadness - (kGoodFrequencies[i]
                    - kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                stDevRelativeGoodMinusBadness += (meanRelativeGoodMinusBadness
                        - ((kGoodFrequencies[i] - kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (meanRelativeGoodMinusBadness - ((kGoodFrequencies[i]
                        - kBadFrequencies[i]) / kNeighborFrequencies[i]));
            } else {
                stDevRelativeGoodMinusBadness +=
                        (meanRelativeGoodMinusBadness - 1)
                        * (meanRelativeGoodMinusBadness - 1);
            }
        }
        // Normalize the averages.
        stDevOccBadness /= (float) kBadFrequencies.length;
        stDevOccGoodness /= (float) kGoodFrequencies.length;
        stDevGoodMinusBadness /= (float) kGoodFrequencies.length;
        stDevRelativeGoodMinusBadness /= (float) kGoodFrequencies.length;
        // Take the square root of the variances to obtain the standard
        // deviations.
        stDevOccBadness = Math.sqrt(stDevOccBadness);
        stDevOccGoodness = Math.sqrt(stDevOccGoodness);
        stDevGoodMinusBadness = Math.sqrt(stDevGoodMinusBadness);
        stDevRelativeGoodMinusBadness =
                Math.sqrt(stDevRelativeGoodMinusBadness);
    }

    /**
     * @return float[] The error-inducing neighbor occurrence counts.
     */
    public float[] getErrorInducingHubness() {
        return getErrorInducingHubness(kNeighbors[0].length);
    }

    /**
     * Calculates the error-inducing hubness counts, by counting how many of the
     * bad occurrences contribute to actual misclassification.
     *
     * @param k Integer that is the neighborhood size.
     * @param dataLabels int[] representing the external data labels.
     * @return The error-inducing neighbor occurrence counts.
     */
    public float[] getErrorInducingHubness(int k, int[] dataLabels) {
        int len = kNeighbors.length;
        int numClasses = ArrayUtil.max(dataLabels) + 1;
        float[] errOccFreqs = new float[len];
        float[] classCounts = new float[numClasses];
        float maxClassVote;
        int maxClassIndex;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < k; j++) {
                classCounts[dataLabels[kNeighbors[i][j]]]++;
            }
            maxClassVote = 0;
            maxClassIndex = 0;
            for (int c = 0; c < numClasses; c++) {
                if (classCounts[c] > maxClassVote) {
                    maxClassVote = classCounts[c];
                    maxClassIndex = c;
                }
            }
            if (maxClassIndex != dataLabels[i]) {
                // An error-inducing occurrence, as it would have contributed
                // to misclassification here.
                for (int j = 0; j < k; j++) {
                    if (dataLabels[i] != dataLabels[kNeighbors[i][j]]) {
                        errOccFreqs[kNeighbors[i][j]]++;
                    }
                }
            }
        }
        return errOccFreqs;
    }

    /**
     * Calculates the error-inducing hubness counts, by counting how many of the
     * bad occurrences contribute to actual misclassification.
     *
     * @param k Integer that is the neighborhood size.
     * @return The error-inducing neighbor occurrence counts.
     */
    public float[] getErrorInducingHubness(int k) {
        int len = kNeighbors.length;
        int numClasses = dset.countCategories();
        float[] errOccFreqs = new float[len];
        float[] classCounts = new float[numClasses];
        float maxClassVote;
        int maxClassIndex;
        for (int i = 0; i < len; i++) {
            for (int j = 0; j < k; j++) {
                classCounts[dset.getLabelOf(kNeighbors[i][j])]++;
            }
            maxClassVote = 0;
            maxClassIndex = 0;
            for (int c = 0; c < numClasses; c++) {
                if (classCounts[c] > maxClassVote) {
                    maxClassVote = classCounts[c];
                    maxClassIndex = c;
                }
            }
            if (maxClassIndex != dset.getLabelOf(i)) {
                // An error-inducing occurrence, as it would have contributed
                // to misclassification here.
                for (int j = 0; j < k; j++) {
                    if (dset.getLabelOf(i)
                            != dset.getLabelOf(kNeighbors[i][j])) {
                        errOccFreqs[kNeighbors[i][j]]++;
                    }
                }
            }
        }
        return errOccFreqs;
    }

    /**
     * @return DataInstance that is the major hub in the dataset.
     */
    public DataInstance getMajorHubInstance() {
        return dset.data.get(getHubIndex());
    }

    /**
     * @return Integer that is the index of the major hub in the dataset.
     */
    public int getHubIndex() {
        int maxFreq = 0;
        int maxIndex = -1;
        for (int i = 0; i < kNeighborFrequencies.length; i++) {
            if (kNeighborFrequencies[i] > maxFreq) {
                maxFreq = kNeighborFrequencies[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * @return CombinedMetric object for distance calculations.
     */
    public CombinedMetric getCombinedMetric() {
        return cmet;
    }

    /**
     * This method constructs a NeighborSetFinder object from the approximate
     * NeighborSetFinder implementation via Lanczos bisections.
     *
     * @param appNSF AppKNNGraphLanczosBisection approximate kNN implementation.
     * @param calculateMissingDistances Boolean flag indicating whether to
     * calculate the missing distances or not.
     * @return NeighborSetFinder object based on the kNN sets in the approximate
     * kNN implementation.
     * @throws Exception
     */
    public static NeighborSetFinder constructFromAppFinder(
            AppKNNGraphLanczosBisection appNSF,
            boolean calculateMissingDistances) throws Exception {
        NeighborSetFinder nsf = new NeighborSetFinder();
        nsf.dset = appNSF.getDataSet();
        nsf.currK = appNSF.getK();
        nsf.distMatrix = appNSF.getDistances();
        CombinedMetric cmet = appNSF.getMetric();
        // Calculate the distance matrix, if specified.
        if (calculateMissingDistances) {
            for (int i = 0; i < nsf.dset.size(); i++) {
                for (int j = 0; j < nsf.distMatrix[i].length; j++) {
                    if (!appNSF.getDistanceFlags()[i][j]) {
                        nsf.distMatrix[i][j] = cmet.dist(nsf.dset.data.get(i),
                                nsf.dset.data.get(i + j + 1));
                    }
                }
            }
        }
        nsf.cmet = cmet;
        // Initialize the reverse neighbor lists.
        nsf.reverseNeighbors = new ArrayList[nsf.dset.size()];
        for (int i = 0; i < nsf.reverseNeighbors.length; i++) {
            nsf.reverseNeighbors[i] = new ArrayList<>(appNSF.getK() * 4);
        }
        nsf.distancesCalculated = true;
        // Get the kNN sets and the k-distances.
        nsf.kDistances = appNSF.getKdistances();
        nsf.kNeighbors = appNSF.getKneighbors();
        nsf.kNeighborFrequencies = new int[nsf.kNeighbors.length];
        nsf.kBadFrequencies = new int[nsf.kNeighbors.length];
        nsf.kGoodFrequencies = new int[nsf.kNeighbors.length];
        // Fill in the reverse neighbor lists and the occurrence frequency
        // counts.
        for (int i = 0; i < nsf.kNeighbors.length; i++) {
            for (int j = 0; j < appNSF.getK(); j++) {
                nsf.reverseNeighbors[nsf.kNeighbors[i][j]].add(i);
                nsf.kNeighborFrequencies[nsf.kNeighbors[i][j]]++;
                if (nsf.dset.data.get(i).getCategory() != nsf.dset.data.get(
                        nsf.kNeighbors[i][j]).getCategory()) {
                    nsf.kBadFrequencies[nsf.kNeighbors[i][j]]++;
                } else {
                    nsf.kGoodFrequencies[nsf.kNeighbors[i][j]]++;
                }
            }
        }
        // Calculate the occurrence frequency stats.
        nsf.meanOccFreq = 0;
        nsf.stDevOccFreq = 0;
        nsf.meanOccBadness = 0;
        nsf.stDevOccBadness = 0;
        nsf.meanOccGoodness = 0;
        nsf.stDevOccGoodness = 0;
        nsf.meanGoodMinusBadness = 0;
        nsf.stDevGoodMinusBadness = 0;
        nsf.meanRelativeGoodMinusBadness = 0;
        nsf.stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < nsf.kBadFrequencies.length; i++) {
            nsf.meanOccFreq += nsf.kNeighborFrequencies[i];
            nsf.meanOccBadness += nsf.kBadFrequencies[i];
            nsf.meanOccGoodness += nsf.kGoodFrequencies[i];
            nsf.meanGoodMinusBadness += nsf.kGoodFrequencies[i]
                    - nsf.kBadFrequencies[i];
            if (nsf.kNeighborFrequencies[i] > 0) {
                nsf.meanRelativeGoodMinusBadness += ((nsf.kGoodFrequencies[i]
                        - nsf.kBadFrequencies[i])
                        / nsf.kNeighborFrequencies[i]);
            } else {
                nsf.meanRelativeGoodMinusBadness += 1;
            }
        }
        nsf.meanOccFreq /= (float) nsf.kNeighborFrequencies.length;
        nsf.meanOccBadness /= (float) nsf.kBadFrequencies.length;
        nsf.meanOccGoodness /= (float) nsf.kGoodFrequencies.length;
        nsf.meanGoodMinusBadness /= (float) nsf.kGoodFrequencies.length;
        nsf.meanRelativeGoodMinusBadness /= (float) nsf.kGoodFrequencies.length;
        for (int i = 0; i < nsf.kBadFrequencies.length; i++) {
            nsf.stDevOccFreq += ((nsf.meanOccFreq
                    - nsf.kNeighborFrequencies[i])
                    * (nsf.meanOccFreq - nsf.kNeighborFrequencies[i]));
            nsf.stDevOccBadness += ((nsf.meanOccBadness
                    - nsf.kBadFrequencies[i]) * (nsf.meanOccBadness
                    - nsf.kBadFrequencies[i]));
            nsf.stDevOccGoodness += ((nsf.meanOccGoodness
                    - nsf.kGoodFrequencies[i]) * (nsf.meanOccGoodness
                    - nsf.kGoodFrequencies[i]));
            nsf.stDevGoodMinusBadness += ((nsf.meanGoodMinusBadness
                    - (nsf.kGoodFrequencies[i] - nsf.kBadFrequencies[i]))
                    * (nsf.meanGoodMinusBadness - (nsf.kGoodFrequencies[i]
                    - nsf.kBadFrequencies[i])));
            if (nsf.kNeighborFrequencies[i] > 0) {
                nsf.stDevRelativeGoodMinusBadness +=
                        (nsf.meanRelativeGoodMinusBadness
                        - ((nsf.kGoodFrequencies[i] - nsf.kBadFrequencies[i])
                        / nsf.kNeighborFrequencies[i]))
                        * (nsf.meanRelativeGoodMinusBadness
                        - ((nsf.kGoodFrequencies[i] - nsf.kBadFrequencies[i])
                        / nsf.kNeighborFrequencies[i]));
            } else {
                nsf.stDevRelativeGoodMinusBadness +=
                        (nsf.meanRelativeGoodMinusBadness - 1)
                        * (nsf.meanRelativeGoodMinusBadness - 1);
            }
        }
        nsf.stDevOccFreq /= (float) nsf.kNeighborFrequencies.length;
        nsf.stDevOccBadness /= (float) nsf.kBadFrequencies.length;
        nsf.stDevOccGoodness /= (float) nsf.kGoodFrequencies.length;
        nsf.stDevGoodMinusBadness /= (float) nsf.kGoodFrequencies.length;
        nsf.stDevRelativeGoodMinusBadness /=
                (float) nsf.kGoodFrequencies.length;
        nsf.stDevOccFreq = Math.sqrt(nsf.stDevOccFreq);
        nsf.stDevOccBadness = Math.sqrt(nsf.stDevOccBadness);
        nsf.stDevOccGoodness = Math.sqrt(nsf.stDevOccGoodness);
        nsf.stDevGoodMinusBadness = Math.sqrt(nsf.stDevGoodMinusBadness);
        nsf.stDevRelativeGoodMinusBadness =
                Math.sqrt(nsf.stDevRelativeGoodMinusBadness);
        return nsf;
    }

    /**
     * @return ArrayList<Integer>[] that is an array of reverse neighbor lists
     * for all points in the data.
     */
    public ArrayList<Integer>[] getReverseNeighbors() {
        return reverseNeighbors;
    }

    public float getAvgDistToNNposition(int kPos) {
        int indexPos = kPos - 1;
        double sum = 0;
        for (int i = 0; i < kNeighbors.length; i++) {
            sum += kDistances[i][indexPos];
        }
        float result = (float) (sum / (double) kNeighbors.length);
        return result;
    }

    /**
     * Calculates and generates a NeighborSetFinder object that represents a
     * smaller k-range than the current object, on the prototype restriction in
     * case of instance selection.
     *
     * @param kSmaller Integer that is the neighborhood size to calculate the
     * kNN sets for.
     * @param prototypeIndexes ArrayList<Integer> of selected prototype indexes.
     * @param protoDistances float[][] representing the distance matrix on the
     * prototype set.
     * @param protoDSet DataSet that is the prototype data context.
     * @return NeighborSetFinder that is the calculated restriction.
     * @throws Exception
     */
    public NeighborSetFinder getSubNSF(int kSmaller,
            ArrayList<Integer> prototypeIndexes, float[][] protoDistances,
            DataSet protoDSet) throws Exception {
        // Initialize the prototype index maps.
        HashMap<Integer, Integer> originalIndexMap =
                new HashMap<>(prototypeIndexes.size() * 2);
        HashMap<Integer, Integer> protoMap =
                new HashMap<>(prototypeIndexes.size() * 2);
        int protoSize = prototypeIndexes.size();
        for (int i = 0; i < protoSize; i++) {
            originalIndexMap.put(i, prototypeIndexes.get(i));
            protoMap.put(prototypeIndexes.get(i), i);
        }
        // Initialize the resulting restriction.
        NeighborSetFinder nsfRestriction = new NeighborSetFinder(
                protoDSet, protoDistances, cmet);
        nsfRestriction.kNeighbors = new int[protoSize][kSmaller];
        nsfRestriction.kDistances = new float[protoSize][kSmaller];
        nsfRestriction.kCurrLen = new int[protoSize];
        nsfRestriction.kNeighborFrequencies = new int[protoSize];
        nsfRestriction.kBadFrequencies = new int[protoSize];
        nsfRestriction.kGoodFrequencies = new int[protoSize];
        // Intervals used for quick restriction calculations in the kNN sets.
        ArrayList<Integer> knnSetIntervals;
        int upperIndex, lowerIndex;
        int minIndVal, maxIndVal;
        // Neighborhood size.
        int k;
        // Auxiliary variable for kNN search.
        int l;
        for (int i = 0; i < protoSize; i++) {
            k = kCurrLen[prototypeIndexes.get(i)];
            nsfRestriction.kCurrLen[i] = 0;
            knnSetIntervals = new ArrayList(kSmaller + 2);
            knnSetIntervals.add(-1);
            for (int j = 0; j < k; j++) {
                if (protoMap.containsKey(
                        kNeighbors[prototypeIndexes.get(i)][j])) {
                    nsfRestriction.kNeighbors[i][nsfRestriction.kCurrLen[i]] =
                            protoMap.get(
                            kNeighbors[prototypeIndexes.get(i)][j]);
                    nsfRestriction.kDistances[i][nsfRestriction.kCurrLen[i]] =
                            kDistances[prototypeIndexes.get(i)][j];
                    knnSetIntervals.add(
                            nsfRestriction.kNeighbors[i][
                            nsfRestriction.kCurrLen[i]]);
                    nsfRestriction.kCurrLen[i]++;
                }
                if (nsfRestriction.kCurrLen[i] >= kSmaller) {
                    break;
                }
            }
            knnSetIntervals.add(protoSize + 1);
            Collections.sort(knnSetIntervals);
            if (nsfRestriction.kCurrLen[i] < kSmaller) {
                int iSizeRed = knnSetIntervals.size() - 1;
                for (int ind = 0; ind < iSizeRed; ind++) {
                    lowerIndex = knnSetIntervals.get(ind);
                    upperIndex = knnSetIntervals.get(ind + 1);
                    for (int j = lowerIndex + 1; j < upperIndex - 1; j++) {
                        if (i != j) {
                            minIndVal = Math.min(i, j);
                            maxIndVal = Math.max(i, j);

                            if (nsfRestriction.kCurrLen[i] > 0) {
                                if (nsfRestriction.kCurrLen[i] == kSmaller) {
                                    if (protoDistances[minIndVal][
                                            maxIndVal - minIndVal - 1]
                                            < nsfRestriction.kDistances[i][
                                            nsfRestriction.kCurrLen[i] - 1]) {
                                        // Search and insert.
                                        l = kSmaller - 1;
                                        while ((l >= 1) && protoDistances[
                                                minIndVal][maxIndVal - minIndVal
                                                - 1]
                                                < nsfRestriction.kDistances[i][
                                                l - 1]) {
                                            nsfRestriction.kDistances[i][l] =
                                                    nsfRestriction.kDistances[
                                                    i][l - 1];
                                            nsfRestriction.kNeighbors[i][l] =
                                                    nsfRestriction.kNeighbors[
                                                    i][l - 1];
                                            l--;
                                        }
                                        nsfRestriction.kDistances[i][l] =
                                                protoDistances[minIndVal][
                                                maxIndVal - minIndVal - 1];
                                        nsfRestriction.kNeighbors[i][l] = j;
                                    }
                                } else {
                                    if (protoDistances[minIndVal][maxIndVal
                                            - minIndVal - 1]
                                            < nsfRestriction.kDistances[i][
                                            nsfRestriction.kCurrLen[i] - 1]) {
                                        // Search and insert.
                                        l = nsfRestriction.kCurrLen[i] - 1;
                                        nsfRestriction.kDistances[i][
                                                nsfRestriction.kCurrLen[i]] =
                                                nsfRestriction.kDistances[i][
                                                nsfRestriction.kCurrLen[i] - 1];
                                        nsfRestriction.kNeighbors[i][
                                                nsfRestriction.kCurrLen[i]] =
                                                nsfRestriction.kNeighbors[i][
                                                nsfRestriction.kCurrLen[i] - 1];
                                        while ((l >= 1) && protoDistances[
                                                minIndVal][maxIndVal - minIndVal
                                                - 1]
                                                < nsfRestriction.kDistances[i][
                                                l - 1]) {
                                            nsfRestriction.kDistances[i][l] =
                                                    nsfRestriction.kDistances[
                                                    i][l - 1];
                                            nsfRestriction.kNeighbors[i][l] =
                                                    nsfRestriction.kNeighbors[
                                                    i][l - 1];
                                            l--;
                                        }
                                        nsfRestriction.kDistances[i][l] =
                                                protoDistances[minIndVal][
                                                maxIndVal - minIndVal - 1];
                                        nsfRestriction.kNeighbors[i][l] = j;
                                        nsfRestriction.kCurrLen[i]++;
                                    } else {
                                        nsfRestriction.kDistances[i][
                                                nsfRestriction.kCurrLen[i]] =
                                                protoDistances[minIndVal][
                                                maxIndVal - minIndVal - 1];
                                        nsfRestriction.kNeighbors[i][
                                                nsfRestriction.kCurrLen[i]] = j;
                                        nsfRestriction.kCurrLen[i]++;
                                    }
                                }
                            } else {
                                nsfRestriction.kDistances[i][0] =
                                        protoDistances[minIndVal][maxIndVal
                                        - minIndVal - 1];
                                nsfRestriction.kNeighbors[i][0] = j;
                                nsfRestriction.kCurrLen[i] = 1;
                            }
                        }
                    }
                }
            }
        }
        nsfRestriction.currK = kSmaller;
        // Calculate the neighbor occurrence frequencies and the reverse
        // neighbor sets for the calculated restriction.
        int currClass;
        int nClass;
        nsfRestriction.reverseNeighbors = new ArrayList[protoSize];
        for (int i = 0; i < protoSize; i++) {
            nsfRestriction.reverseNeighbors[i] = new ArrayList<>(10 * kSmaller);
        }
        for (int i = 0; i < protoSize; i++) {
            currClass = protoDSet.getLabelOf(i);

            for (int j = 0; j < kSmaller; j++) {
                nClass = protoDSet.getLabelOf(nsfRestriction.kNeighbors[i][j]);
                nsfRestriction.reverseNeighbors[
                        nsfRestriction.kNeighbors[i][j]].add(i);
                if (nClass == currClass) {
                    nsfRestriction.kGoodFrequencies[
                            nsfRestriction.kNeighbors[i][j]]++;
                } else {
                    nsfRestriction.kBadFrequencies[
                            nsfRestriction.kNeighbors[i][j]]++;
                }
                nsfRestriction.kNeighborFrequencies[
                        nsfRestriction.kNeighbors[i][j]]++;
            }
        }

        nsfRestriction.completeNeighborSets(kSmaller, null);
        return nsfRestriction;
    }

    /**
     * Calculates and generates a NeighborSetFinder object that represents a
     * smaller k-range than the current object.
     *
     * @param kSmaller Integer that is the neighborhood size to re-calculate the
     * kNN sets for.
     * @return NeighborSetFinder that is the restriction on a smaller
     * neighborhood size.
     */
    public NeighborSetFinder getSubNSF(int kSmaller) {
        NeighborSetFinder nsfRestriction =
                new NeighborSetFinder(dset, distMatrix, cmet);
        nsfRestriction.kNeighbors = new int[dset.size()][];
        nsfRestriction.kDistances = new float[dset.size()][];
        nsfRestriction.kCurrLen = new int[dset.size()];
        // Copy the full occurrence counts - we will subtract the counts for
        // the difference between the k values below.
        nsfRestriction.kNeighborFrequencies =
                Arrays.copyOfRange(kNeighborFrequencies, 0, dset.size());
        nsfRestriction.kBadFrequencies =
                Arrays.copyOfRange(kBadFrequencies, 0, dset.size());
        nsfRestriction.kGoodFrequencies =
                Arrays.copyOfRange(kGoodFrequencies, 0, dset.size());
        for (int i = 0; i < dset.size(); i++) {
            nsfRestriction.kNeighbors[i] =
                    Arrays.copyOfRange(kNeighbors[i], 0, kSmaller);
            nsfRestriction.kDistances[i] =
                    Arrays.copyOfRange(kDistances[i], 0, kSmaller);
        }
        Arrays.fill(kCurrLen, kSmaller);
        nsfRestriction.currK = kSmaller;
        nsfRestriction.reverseNeighbors = new ArrayList[dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            nsfRestriction.reverseNeighbors[i] = new ArrayList<>(10 * kSmaller);
        }
        for (int i = 0; i < dset.size(); i++) {
            for (int j = 0; j < kSmaller; j++) {
                nsfRestriction.reverseNeighbors[kNeighbors[i][j]].add(i);
            }
        }
        // Subtract the additional occurrence counts in the k-range between the
        // two k values.
        for (int i = 0; i < dset.size(); i++) {
            for (int j = kSmaller; j < kNeighbors[0].length; j++) {
                nsfRestriction.kNeighborFrequencies[kNeighbors[i][j]]--;
                if (dset.data.get(i).getCategory()
                        == dset.data.get(kNeighbors[i][j]).getCategory()) {
                    nsfRestriction.kGoodFrequencies[kNeighbors[i][j]]--;
                } else {
                    nsfRestriction.kBadFrequencies[kNeighbors[i][j]]--;
                }
            }
        }
        // Calculate the neighbor occurrence frequency stats.
        nsfRestriction.meanOccFreq = 0;
        nsfRestriction.stDevOccFreq = 0;
        nsfRestriction.meanOccBadness = 0;
        nsfRestriction.stDevOccBadness = 0;
        nsfRestriction.meanOccGoodness = 0;
        nsfRestriction.stDevOccGoodness = 0;
        nsfRestriction.meanGoodMinusBadness = 0;
        nsfRestriction.meanRelativeGoodMinusBadness = 0;
        nsfRestriction.stDevGoodMinusBadness = 0;
        nsfRestriction.stDevRelativeGoodMinusBadness = 0;
        for (int i = 0; i < nsfRestriction.kBadFrequencies.length; i++) {
            nsfRestriction.meanOccFreq +=
                    nsfRestriction.kNeighborFrequencies[i];
            nsfRestriction.meanOccBadness += nsfRestriction.kBadFrequencies[i];
            nsfRestriction.meanOccGoodness +=
                    nsfRestriction.kGoodFrequencies[i];
            nsfRestriction.meanGoodMinusBadness +=
                    nsfRestriction.kGoodFrequencies[i]
                    - nsfRestriction.kBadFrequencies[i];
            if (kNeighborFrequencies[i] > 0) {
                nsfRestriction.meanRelativeGoodMinusBadness +=
                        ((nsfRestriction.kGoodFrequencies[i]
                        - nsfRestriction.kBadFrequencies[i])
                        / kNeighborFrequencies[i]);
            } else {
                nsfRestriction.meanRelativeGoodMinusBadness += 1;
            }
        }
        nsfRestriction.meanOccFreq /=
                (float) nsfRestriction.kNeighborFrequencies.length;
        nsfRestriction.meanOccBadness /=
                (float) nsfRestriction.kBadFrequencies.length;
        nsfRestriction.meanOccGoodness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        nsfRestriction.meanGoodMinusBadness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        nsfRestriction.meanRelativeGoodMinusBadness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        for (int i = 0; i < nsfRestriction.kBadFrequencies.length; i++) {
            nsfRestriction.stDevOccFreq +=
                    ((nsfRestriction.meanOccFreq
                    - nsfRestriction.kNeighborFrequencies[i])
                    * (nsfRestriction.meanOccFreq
                    - nsfRestriction.kNeighborFrequencies[i]));
            nsfRestriction.stDevOccBadness +=
                    ((nsfRestriction.meanOccBadness
                    - nsfRestriction.kBadFrequencies[i])
                    * (nsfRestriction.meanOccBadness
                    - nsfRestriction.kBadFrequencies[i]));
            nsfRestriction.stDevOccGoodness +=
                    ((nsfRestriction.meanOccGoodness
                    - nsfRestriction.kGoodFrequencies[i])
                    * (nsfRestriction.meanOccGoodness
                    - nsfRestriction.kGoodFrequencies[i]));
            nsfRestriction.stDevGoodMinusBadness +=
                    ((nsfRestriction.meanGoodMinusBadness
                    - (nsfRestriction.kGoodFrequencies[i]
                    - nsfRestriction.kBadFrequencies[i]))
                    * (nsfRestriction.meanGoodMinusBadness
                    - (nsfRestriction.kGoodFrequencies[i]
                    - nsfRestriction.kBadFrequencies[i])));
            if (kNeighborFrequencies[i] > 0) {
                nsfRestriction.stDevRelativeGoodMinusBadness +=
                        (nsfRestriction.meanRelativeGoodMinusBadness
                        - ((nsfRestriction.kGoodFrequencies[i]
                        - nsfRestriction.kBadFrequencies[i])
                        / kNeighborFrequencies[i]))
                        * (nsfRestriction.meanRelativeGoodMinusBadness
                        - ((nsfRestriction.kGoodFrequencies[i]
                        - nsfRestriction.kBadFrequencies[i])
                        / kNeighborFrequencies[i]));
            } else {
                nsfRestriction.stDevRelativeGoodMinusBadness +=
                        (nsfRestriction.meanRelativeGoodMinusBadness)
                        * (nsfRestriction.meanRelativeGoodMinusBadness - 1);
            }
        }
        // Normalize the averages.
        nsfRestriction.stDevOccFreq /=
                (float) nsfRestriction.kNeighborFrequencies.length;
        nsfRestriction.stDevOccBadness /=
                (float) nsfRestriction.kBadFrequencies.length;
        nsfRestriction.stDevOccGoodness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        nsfRestriction.stDevGoodMinusBadness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        nsfRestriction.stDevRelativeGoodMinusBadness /=
                (float) nsfRestriction.kGoodFrequencies.length;
        // Take the square root of the variances to obtain the
        // standard deviations.
        nsfRestriction.stDevOccFreq =
                Math.sqrt(nsfRestriction.stDevOccFreq);
        nsfRestriction.stDevOccBadness =
                Math.sqrt(nsfRestriction.stDevOccBadness);
        nsfRestriction.stDevOccGoodness =
                Math.sqrt(nsfRestriction.stDevOccGoodness);
        nsfRestriction.stDevGoodMinusBadness =
                Math.sqrt(nsfRestriction.stDevGoodMinusBadness);
        nsfRestriction.stDevRelativeGoodMinusBadness =
                Math.sqrt(nsfRestriction.stDevRelativeGoodMinusBadness);
        return nsfRestriction;
    }

    /**
     * This method calculates the class-to-class neighbor occurrence probability
     * matrix for use in the Bayesian hubness-aware classification models.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * distribution smoothing.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-to-class neighbor occurrence
     * probability matrix for use in the Bayesian hubness-aware classification
     * models.
     */
    public float[][] getGlobalClassToClassForKforBayerisan(int k,
            int numClasses, float laplaceEstimator, boolean extendByElement) {
        float[][] classToClassPriors = new float[numClasses][numClasses];
        float[] classPriors = new float[numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            classPriors[currClass]++;
            if (extendByElement) {
                classToClassPriors[currClass][currClass]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                classToClassPriors[dset.data.get(kNeighbors[i][kInd]).
                        getCategory()][currClass]++;
            }
        }
        float laplaceTotal = numClasses * laplaceEstimator;
        if (extendByElement) {
            for (int cFirst = 0; cFirst < numClasses; cFirst++) {
                for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                    classToClassPriors[cFirst][cSecond] += laplaceEstimator;
                    classToClassPriors[cFirst][cSecond] /= ((k + 1)
                            * classPriors[cSecond] + laplaceTotal);
                }
            }
        } else {
            for (int cFirst = 0; cFirst < numClasses; cFirst++) {
                for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                    classToClassPriors[cFirst][cSecond] += laplaceEstimator;
                    classToClassPriors[cFirst][cSecond] /=
                            (k * classPriors[cSecond] + laplaceTotal);
                }
            }
        }
        return classToClassPriors;
    }

    /**
     * This method calculates the class-to-class neighbor occurrence probability
     * matrix for use in the fuzzy hubness-aware classification models.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * distribution smoothing.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-to-class neighbor occurrence
     * probability matrix for use in the fuzzy hubness-aware classification
     * models.
     */
    public float[][] getGlobalClassToClassForKforFuzzy(int k, int numClasses,
            float laplaceEstimator, boolean extendByElement) {
        float[][] classToClassPriors = new float[numClasses][numClasses];
        float[] classHubnessSums = new float[numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            if (extendByElement) {
                classToClassPriors[currClass][currClass]++;
                classHubnessSums[currClass]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                classToClassPriors[dset.data.get(
                        kNeighbors[i][kInd]).getCategory()][currClass]++;
                classHubnessSums[dset.data.get(
                        kNeighbors[i][kInd]).getCategory()]++;
            }
        }
        float laplaceTotal = numClasses * laplaceEstimator;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                classToClassPriors[cFirst][cSecond] += laplaceEstimator;
                classToClassPriors[cFirst][cSecond] /=
                        (classHubnessSums[cFirst] + laplaceTotal);
            }
        }
        return classToClassPriors;
    }

    /**
     * This method calculates the class-to-class neighbor occurrence probability
     * matrix, non-normalized.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @return float[][] representing the class-to-class neighbor occurrence
     * probability matrix, non-normalized.
     */
    public float[][] getGlobalClassToClassNonNormalized(int k, int numClasses) {
        float[][] classToClass = new float[numClasses][numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            for (int kInd = 0; kInd < k; kInd++) {
                classToClass[dset.data.get(
                        kNeighbors[i][kInd]).getCategory()][currClass]++;
            }
        }
        return classToClass;
    }

    /**
     * This method calculates the class-to-class neighbor occurrence probability
     * matrix for use in the fuzzy hubness-aware classification models,
     * restricted on points that have bad occurrences.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * distribution smoothing.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-to-class neighbor occurrence
     * probability matrix for use in the fuzzy hubness-aware classification
     * models, restricted on points that have bad occurrences.
     */
    public float[][] getGlobalClassToClassForKforFuzzyRestrictOnBad(int k,
            int numClasses, float laplaceEstimator, boolean extendByElement) {
        float[][] classToClassPriors = new float[numClasses][numClasses];
        float[] classHubnessSums = new float[numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            if (extendByElement) {
                classToClassPriors[currClass][currClass]++;
                classHubnessSums[currClass]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                if (kBadFrequencies[kNeighbors[i][kInd]] > 0) {
                    classToClassPriors[dset.data.get(
                            kNeighbors[i][kInd]).getCategory()][currClass]++;
                    classHubnessSums[dset.data.get(
                            kNeighbors[i][kInd]).getCategory()]++;
                }
            }
        }
        float laplaceTotal = numClasses * laplaceEstimator;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < numClasses; cSecond++) {
                classToClassPriors[cFirst][cSecond] += laplaceEstimator;
                classToClassPriors[cFirst][cSecond] /=
                        (classHubnessSums[cFirst] + laplaceTotal);
            }
        }
        return classToClassPriors;
    }

    /**
     * This method calculates the class-conditional neighbor occurrence
     * probability matrix for use in the Bayesian hubness-aware classification
     * models. This means that the values are normalized to represent the
     * probability of neighbor given the class, instead of class given the
     * neighbor.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * distribution smoothing.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-conditional neighbor occurrence
     * probability matrix for use in the Bayesian hubness-aware classification
     * models.
     */
    public float[][] getClassDataNeighborRelationForKforBayesian(int k,
            int numClasses, float laplaceEstimator, boolean extendByElement) {
        float[][] classDataKNeighborRelation =
                new float[numClasses][dset.size()];
        float[] classPriors = new float[numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            classPriors[currClass]++;
            if (extendByElement) {
                classDataKNeighborRelation[currClass][i]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                classDataKNeighborRelation[currClass][kNeighbors[i][kInd]]++;
            }
        }
        float laplaceTotal = dset.size() * laplaceEstimator;
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < dset.size(); cSecond++) {
                classDataKNeighborRelation[cFirst][cSecond] += laplaceEstimator;
                classDataKNeighborRelation[cFirst][cSecond] /=
                        (k * classPriors[cFirst] + laplaceTotal);
            }
        }
        return classDataKNeighborRelation;
    }

    /**
     * This method calculates the class-conditional neighbor occurrence
     * probability matrix for use in the fuzzy hubness-aware classification
     * models. This means that the values are normalized to represent the
     * probability of neighbor given the class, instead of class given the
     * neighbor.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * distribution smoothing.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-conditional neighbor occurrence
     * probability matrix for use in the fuzzy hubness-aware classification
     * models.
     */
    public float[][] getFuzzyClassDataNeighborRelation(int k, int numClasses,
            float laplaceEstimator, boolean extendByElement) {
        float[][] classDataKNeighborRelation =
                new float[numClasses][dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            if (extendByElement) {
                classDataKNeighborRelation[currClass][i]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                classDataKNeighborRelation[currClass][kNeighbors[i][kInd]]++;
            }
        }
        float laplaceTotal = numClasses * laplaceEstimator;
        recalculateStatsForSmallerK(k);
        int[] neighbOccFreqs = getNeighborFrequencies();
        for (int cFirst = 0; cFirst < numClasses; cFirst++) {
            for (int cSecond = 0; cSecond < dset.size(); cSecond++) {
                classDataKNeighborRelation[cFirst][cSecond] += laplaceEstimator;
                classDataKNeighborRelation[cFirst][cSecond] /=
                        (neighbOccFreqs[cSecond] + 1 + laplaceTotal);
            }
        }
        recalculateStatsForSmallerK(kNeighbors[0].length);
        return classDataKNeighborRelation;
    }

    /**
     * This method calculates the class-conditional neighbor occurrence counts.
     * In the results, the cell [i,j] holds the occurrence count of point i in
     * class j.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-conditional neighbor occurrence
     * counts. In the results, the cell [i,j] holds the occurrence count of
     * point i in class j.
     */
    public float[][] getDataClassNeighborRelationNonNormalized(int k,
            int numClasses, boolean extendByElement) {
        float[][] dataClassKNeighborRelation =
                new float[dset.size()][numClasses];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            if (extendByElement) {
                dataClassKNeighborRelation[i][currClass]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                dataClassKNeighborRelation[kNeighbors[i][kInd]][currClass]++;
            }
        }
        return dataClassKNeighborRelation;
    }

    /**
     * This method calculates the class-conditional neighbor occurrence counts.
     * In the results, the cell [i,j] holds the occurrence count of point j in
     * class i.
     *
     * @param k Integer that is the neighborhood size.
     * @param numClasses Integer that is the number of classes in the data.
     * @param extendByElement Boolean flag indicating whether to use the query
     * point as its own 0-th nearest neighbor.
     * @return float[][] representing the class-conditional neighbor occurrence
     * counts. In the results, the cell [i,j] holds the occurrence count of
     * point j in class i.
     */
    public float[][] getClassDataNeighborRelationNonNormalized(int k,
            int numClasses, boolean extendByElement) {
        float[][] classDataKNeighborRelation =
                new float[numClasses][dset.size()];
        for (int i = 0; i < dset.size(); i++) {
            int currClass = dset.data.get(i).getCategory();
            if (extendByElement) {
                classDataKNeighborRelation[currClass][i]++;
            }
            for (int kInd = 0; kInd < k; kInd++) {
                classDataKNeighborRelation[currClass][kNeighbors[i][kInd]]++;
            }
        }
        return classDataKNeighborRelation;
    }

    /**
     * @return float[][] representing an array of occurrence frequency arrays,
     * for neighborhood sizes from 1 to the maximum neighborhood size supported
     * by the current kNN sets, which is their length.
     */
    public float[][] getOccFreqsForAllK() {
        int k = kNeighbors[0].length;
        float[][] occFreqsAllK = new float[k][kNeighbors.length];
        for (int kInd = 0; kInd < k; kInd++) {
            recalculateStatsForSmallerK(kInd + 1);
            for (int i = 0; i < kNeighbors.length; i++) {
                occFreqsAllK[kInd][i] = getNeighborFrequencies()[i];
            }
        }
        return occFreqsAllK;
    }

    /**
     * This method queries the dataset to determine the neighbors of a
     * particular point from the dataset.
     *
     * @param dset DataSet object to query.
     * @param instanceIndex Integer that is the index of the data instance that
     * is the kNN query.
     * @param neighborhoodSize Integer that is the desired neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @return int[] that contains the indexes of the k-nearest neighbors for
     * the query point.
     * @throws Exception
     */
    public static int[] getIndexesOfNeighbors(DataSet dset, int instanceIndex,
            int neighborhoodSize, CombinedMetric cmet) throws Exception {
        int[] neighbors = new int[neighborhoodSize];
        float currDist;
        int kCurrLen = 0;
        float[] nDists = new float[neighborhoodSize];
        Arrays.fill(nDists, Float.MAX_VALUE);
        int l;
        DataInstance instance = dset.data.get(instanceIndex);
        // Check the first half of the points, with the index value below the
        // query index value.
        for (int i = 0; i < instanceIndex; i++) {
            currDist = cmet.dist(instance, dset.data.get(i));
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and inser.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;

                    }
                } else {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && currDist < nDists[l]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l + 1] = currDist;
                        neighbors[l + 1] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = currDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = currDist;
                nDists[0] = i;
                kCurrLen = 1;
            }
        }
        // Check the second half of the points, with the index value above the
        // query index value.
        for (int i = instanceIndex + 1; i < dset.size(); i++) {
            currDist = cmet.dist(instance, dset.data.get(i));
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                    }
                } else {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = currDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = currDist;
                neighbors[0] = i;
                kCurrLen = 1;
            }
        }
        return neighbors;
    }

    /**
     * This method queries the dataset to determine the neighbors of a
     * particular point that does not belong to the dataset.
     *
     * @param dset DataSet object to query.
     * @param instance DataInstance that is the query point.
     * @param neighborhoodSize Integer that is the desired neighborhood size.
     * @param distances float[] representing the distances from the query point
     * to the points in the provided DataSet.
     * @return int[] that contains the indexes of the k-nearest neighbors for
     * the query point.
     * @throws Exception
     */
    public static int[] getIndexesOfNeighbors(DataSet dset,
            DataInstance instance, int neighborhoodSize, float[] distances)
            throws Exception {
        int[] neighbors = new int[neighborhoodSize];
        float currDist;
        int kCurrLen = 0;
        float[] nDists = new float[neighborhoodSize];
        Arrays.fill(nDists, Float.MAX_VALUE);
        int l;
        for (int i = 0; i < dset.size(); i++) {
            currDist = distances[i];
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                    }
                } else {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = currDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = currDist;
                neighbors[0] = i;
                kCurrLen = 1;
            }
        }
        return neighbors;
    }

    /**
     * Get the indexes of neighbors from the training set for all the instances
     * in the test set.
     *
     * @param trainingDSet DataSet object that is the training data.
     * @param testDSet DataSet object that is the test data.
     * @param neighborhoodSize Integer that is the neighborhood size.
     * @param testToTrainDist Float 2D array of distances from test to training.
     * @return Integer 2D array of indexes of neighbors from the training data
     * of points in the test data.
     * @throws Exception
     */
    public static int[][] getIndexesOfNeighbors(DataSet trainingDSet,
            DataSet testDSet, int neighborhoodSize, float[][] testToTrainDist)
            throws Exception {
        if (trainingDSet == null || testDSet == null) {
            return null;
        }
        int[][] neighborIndexes = new int[testDSet.size()][];
        for (int i = 0; i < testDSet.size(); i++) {
            DataInstance instance = testDSet.getInstance(i);
            neighborIndexes[i] = getIndexesOfNeighbors(trainingDSet,
                    instance, neighborhoodSize, testToTrainDist[i]);
        }
        return neighborIndexes;
    }

    /**
     * Get the indexes of neighbors from the training set for all the instances
     * in the test set.
     *
     * @param trainingDSet DataSet object that is the training data.
     * @param testDSet DataSet object that is the test data.
     * @param neighborhoodSize Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @return Integer 2D array of indexes of neighbors from the training data
     * of points in the test data.
     * @throws Exception
     */
    public static int[][] getIndexesOfNeighbors(DataSet trainingDSet,
            DataSet testDSet, int neighborhoodSize, CombinedMetric cmet)
            throws Exception {
        if (trainingDSet == null || testDSet == null) {
            return null;
        }
        int[][] neighborIndexes = new int[testDSet.size()][];
        for (int i = 0; i < testDSet.size(); i++) {
            DataInstance instance = testDSet.getInstance(i);
            neighborIndexes[i] = getIndexesOfNeighbors(trainingDSet,
                    instance, neighborhoodSize, cmet);
        }
        return neighborIndexes;
    }

    /**
     * This method queries the dataset to determine the neighbors of a
     * particular point that does not belong to the dataset.
     *
     * @param dset DataSet object to query.
     * @param instance DataInstance that is the query point.
     * @param neighborhoodSize Integer that is the desired neighborhood size.
     * @param cmet CombinedMetric object for distance calculations. to the
     * points in the provided DataSet.
     * @return int[] that contains the indexes of the k-nearest neighbors for
     * the query point.
     * @throws Exception
     */
    public static int[] getIndexesOfNeighbors(DataSet dset,
            DataInstance instance, int neighborhoodSize, CombinedMetric cmet)
            throws Exception {
        int[] neighbors = new int[neighborhoodSize];
        float currDist;
        int kCurrLen = 0;
        float[] nDists = new float[neighborhoodSize];
        Arrays.fill(nDists, Float.MAX_VALUE);
        int l;
        for (int i = 0; i < dset.size(); i++) {
            currDist = cmet.dist(instance, dset.data.get(i));
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                    }
                } else {
                    if (currDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && currDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = currDist;
                        neighbors[l] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = currDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = currDist;
                neighbors[0] = i;
                kCurrLen = 1;
            }
        }
        return neighbors;
    }

    /**
     * This method queries the dataset with an instance, given a tabu map of
     * points that can not be considered as potential neighbors.
     *
     * @param dset DataSet object to query.
     * @param instance DataInstance that is the query point.
     * @param neighborhoodSize Integer that is the desired neighborhood size.
     * @param cmet CombinedMetric object for distance calculations. to the
     * points in the provided DataSet.
     * @param tabuMap HashMap containing the indexes of points that can not be
     * considered as neighbors currently.
     * @return int[] that contains the indexes of the k-nearest neighbors for
     * the query point.
     * @throws Exception
     */
    public static int[] getIndexesOfNeighbors(DataSet dset,
            DataInstance instance, int neighborhoodSize, CombinedMetric cmet,
            HashMap tabuMap) throws Exception {
        int[] neighbors = new int[neighborhoodSize];
        float tempDist;
        int kCurrLen = 0;
        float[] nDists = new float[neighborhoodSize];
        Arrays.fill(nDists, Float.MAX_VALUE);
        int l;
        for (int i = 0; i < dset.size(); i++) {
            if (tabuMap.containsKey(i)) {
                continue;
            }
            tempDist = cmet.dist(instance, dset.data.get(i));
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (tempDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && tempDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = tempDist;
                        neighbors[l] = i;
                    }
                } else {
                    if (tempDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && tempDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = tempDist;
                        neighbors[l] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = tempDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = tempDist;
                neighbors[0] = i;
                kCurrLen = 1;
            }
        }
        return neighbors;
    }
    
    /**
     * This method queries the dataset with an instance, given a tabu map of
     * points that can not be considered as potential neighbors.
     *
     * @param dMat float[][] Upper triangular distance matrix of the dataset to
     * query.
     * @param instanceIndex Integer that is the instance index.
     * @param neighborhoodSize Integer that is the desired neighborhood size.
     * @param tabuMap HashMap containing the indexes of points that can not be
     * considered as neighbors currently.
     * @return int[] that contains the indexes of the k-nearest neighbors for
     * the query point.
     * @throws Exception
     */
    public static int[] getIndexesOfNeighbors(float[][] dMat, int instanceIndex,
            int neighborhoodSize, HashMap tabuMap) throws Exception {
        int[] neighbors = new int[neighborhoodSize];
        float tempDist;
        int kCurrLen = 0;
        float[] nDists = new float[neighborhoodSize];
        Arrays.fill(nDists, Float.MAX_VALUE);
        int l;
        for (int i = 0; i < dMat.length; i++) {
            if (tabuMap.containsKey(i) || i == instanceIndex) {
                continue;
            }
            int minIndex = Math.min(i, instanceIndex);
            int maxIndex = Math.max(i, instanceIndex);
            tempDist = dMat[minIndex][maxIndex - minIndex - 1];
            if (kCurrLen > 0) {
                if (kCurrLen == neighborhoodSize) {
                    if (tempDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = neighborhoodSize - 1;
                        while ((l >= 1) && tempDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = tempDist;
                        neighbors[l] = i;
                    }
                } else {
                    if (tempDist < nDists[kCurrLen - 1]) {
                        // Search and insert.
                        l = kCurrLen - 1;
                        nDists[kCurrLen] = nDists[kCurrLen - 1];
                        neighbors[kCurrLen] = neighbors[kCurrLen - 1];
                        while ((l >= 1) && tempDist < nDists[l - 1]) {
                            nDists[l] = nDists[l - 1];
                            neighbors[l] = neighbors[l - 1];
                            l--;
                        }
                        nDists[l] = tempDist;
                        neighbors[l] = i;
                        kCurrLen++;
                    } else {
                        nDists[kCurrLen] = tempDist;
                        neighbors[kCurrLen] = i;
                        kCurrLen++;
                    }
                }
            } else {
                nDists[0] = tempDist;
                neighbors[0] = i;
                kCurrLen = 1;
            }
        }
        return neighbors;
    }

    /**
     * @return int[] representing the neighbor occurrence frequencies for all
     * data points.
     */
    public int[] getNeighborFrequencies() {
        return kNeighborFrequencies;
    }

    /**
     * @return int[] that contains the good neighbor occurrence frequencies for
     * all data points.
     */
    public int[] getGoodFrequencies() {
        return kGoodFrequencies;
    }

    /**
     * @return int[] that contains the bad neighbor occurrence frequencies for
     * all data points.
     */
    public int[] getBadFrequencies() {
        return kBadFrequencies;
    }

    /**
     * @return DataSet object that is being analyzed.
     */
    public DataSet getDataSet() {
        return dset;
    }

    /**
     * @return float[] representing the neighbor occurrence frequencies for all
     * data points.
     */
    public float[] getFloatOccFreqs() {
        float[] occFreqs = new float[kNeighbors.length];
        for (int i = 0; i < kNeighbors.length; i++) {
            occFreqs[i] = kNeighborFrequencies[i];
        }
        return occFreqs;
    }

    /**
     * @return float[][] representing the upper triangular distance matrix.
     */
    public float[][] getDistances() {
        return distMatrix;
    }

    /**
     * @return float[][] representing an array of arrays of k-distances for all
     * data points.
     */
    public float[][] getKDistances() {
        return kDistances;
    }

    /**
     * This method calculates an estimate of the data density in all data points
     * based on their nearest neighbors and the kNN radius.
     *
     * @param dset DataSet object that the points belong to.
     * @param cmet CombinedMetric object for distance calculations.
     * @param currK Integer that is the neighborhood size.
     * @param kNeighbors int[][] that is an array of neighbor index arrays for
     * all data points.
     * @return double[] containing the estimated point-wise densities.
     * @throws Excep
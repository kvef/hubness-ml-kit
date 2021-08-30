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
                stDevRelativeGoodMinusBadness += (meanRela
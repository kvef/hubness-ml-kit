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
  
/**
 * Hub Miner: a hubness-aware machine learning experimentation library.
 * Copyright (C) 2014 Nenad Tomasev. Email: nenad.tomasev at gmail.com
 * 
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */
package distances.meta;

import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import ioformat.SupervisedLoader;
import java.util.ArrayList;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * This class implements a recently proposed unsupervised approach for choosing
 * the optimal exponent to use for calculating Minkowski distances in the data.
 * The approach is based on selecting either the exponent leading to the lowest
 * anti-hub rate or the lowest hub rate, as they are correlated. This approach
 * was proposed in the paper titled "Choosing the Metric in High-Dimensional
 * Spaces Based on Hub Analysis" by Dominik Schnitzer and Arthur Flexer that was
 * presented at the 22nd European Symposium on Artificial Neural Networks,
 * Computational Intelligence and Machine Learning in 2014. In that paper, hubs
 * were formally defined as points that occur at least twice as often as
 * expected, which is somewhat less formal and flexible than using the more
 * standard approach of marking points whose occurrence counts exceed the
 * average by at least two standard deviations as hubs. Nevertheless, for
 * consistency with the paper, we use the Nk(x) >= 2k as a criterion for hubs in
 * this class.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MinkowskiDegreeAutoFinder {

    // Whether hub rates or anti-hub rates are to be used in selection.
    public enum DegreeSelectionCriterion {
        
        HUB, ANTIHUB
    }
    private static final int DEFAULT_NUM_THREADS = 8;
    private static final int DEFAULT_NEIGHBORHOOD_SIZE = 1;
    // Data to learn the best exponent for.
    private DataSet dset;
    // The range where to search for the best exponent.
    private float minExp = 0.25f;
    private float maxExp = 4f;
    private float stepExp = 0.25f;
    // The current best distance matrix.
    private float[][] bestMatrix;
    // The current best exponent.
    private float bestExponent;
    // Lists of calculated parameters.
    private ArrayList<Float> testedExponents;
    private ArrayList<Float> antiHubRates;
    private ArrayList<Float> hubRates;
    // The selection criterion.
    private DegreeSelectionCriterion selectionCriterion =
            DegreeSelectionCriterion.ANTIHUB;
    // Number of threads to use for distance matrix calculations.
    private int numThreads = DEFAULT_NUM_THREADS;
    private int k = DEFAULT_NEIGHBORHOOD_SIZE;
    private boolean verbose = false;

    /**
     * Initialization.
     *
     * @param dset DataSet to learn the optimal Minkowski exponent for.
     */
    public MinkowskiDegreeAutoFinder(DataSet dset) {
        if (dset == null || dset.isEmpty()) {
            throw new IllegalArgumentException("DataSet must not be empty.");
        }
        this.dset = dset;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to learn the optimal Minkowski exponent for.
     * @param minExp Float value that is the minimum exponent to try.
     * @param maxExp Float value that is the maximum exponent to try.
     * @param stepExp Float value that is the step for the exponent search.
     */
    public MinkowskiDegreeAutoFinder(DataSet dset, float minExp, float maxExp,
            float stepExp) {
        if (dset == null || dset.isEmpty()) {
            throw new IllegalArgumentException("DataSet must not be empty.");
        }
        this.dset = dset;
        if (minExp > maxEx
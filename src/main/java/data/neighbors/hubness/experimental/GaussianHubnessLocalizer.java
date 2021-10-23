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
package data.neighbors.hubness.experimental;

import data.generators.MultiDimensionalSphericGaussianGenerator;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.concentration.ConcentrationCalculator;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import feature.correlation.DistanceCorrelation;
import feature.correlation.PearsonCorrelation;
import ioformat.FileUtil;
import ioformat.IOARFF;
import java.io.File;
import java.util.ArrayList;

import java.util.Arrays;
import java.util.Random;
import util.CommandLineParser;

/**
 * This class implements an experiment for tracking hub localization in
 * synthetic Gaussian intrinsically high-dimensional data, incrementally. The
 * k-nearest neighbor sets are updated after every new synthetic data instance
 * insertion. The localization is tracked over several different neighborhood
 * sizes and for several metrics - Euclidean, Manhattan and Fractional.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GaussianHubnessLocalizer {

    // Directory for the results.
    private File outDir;
    // Dimensionality of the synthetic data.
    private int dim;
    // Minimal number of instances (when to start the output).
    private int minInst;
    // Maximal number of instances (when to finish the output).
    private int maxInst;
    // The maximal neighborhood size to check.
    private int maxK;
    // The distance matrices for Manhattan, Euclidean and Fractional distances.
    private float[][] distancesMan;
    private float[][] distancesEuc;
    private float[][] distancesFrac;
    // The Manhattan distance auxiliary kNN arrays.
    private int[][] kneighborsMan = null;
    private float[][] kdistancesMan = null;
    private int[] kcurrLenMan = null;
    private float[] kneighborFrequenciesMan = null;
    // The Euclidean distance auxiliary kNN arrays.
    private int[][] kneighborsEuc = null;
    private float[][] kdistancesEuc = null;
    private int[] kcurrLenEuc = null;
    private float[] kneighborFrequenciesEuc = null;
    // The Fractional distance auxiliary kNN arrays.
    private int[][] kneighborsFrac = null;
    private float[][] kdistancesFrac = null;
    private int[] kcurrLenFrac = null;
    private float[] kneighborFrequenciesFrac = null;
    // Cluster medoids for each distance type.
    private DataInstance[] medoidsMan;
    private DataInstance[] medoidsEuc;
    private DataInstance[] medoidsFrac;

    /**
     * Initialization.
     *
     * @param outDir Directory for the output.
     * @param dim Integer that is the desired data dimensionality.
     * @param minInst Integer that is the minimal number of instances.
     * @param maxInst Integer that is the maximal number of instances.
     * @param maxK Integer that is the maximal neighborhood size to consider.
     */
    public GaussianHubnessLocalizer(File outDir, int dim, int minInst,
            int maxInst, int maxK) {
        this.outDir = outDir;
        this.dim = dim;
        this.minInst = minInst;
        this.maxInst = maxInst;
        this.maxK = maxK;
    }

    /**
     * This method runs the entire experiment, incrementally inserting instances
     * into the dataset and updating the kNN sets and tracking for medoid
     * localization in the clusters.
     *
     * @throws Exception
     */
    public void runHubnessLocalizationExperiment() throws Exception {
        // Initialize the distance calculation objects.
        CombinedMetric cmetMan = CombinedMetric.FLOAT_MANHATTAN;
        CombinedMetric cmetEuc = CombinedMetric.FLOAT_EUCLIDEAN;
        CombinedMetric cmetFrac =
                new CombinedMetric(null, new MinkowskiMetric(0.5f),
                CombinedMetric.DEFAULT);
        // Generate the synthetic dataset.
        Random randa = new Random();
        // Initialize the means and the standard deviations.
        float[] featureMeans = new float[dim];
        Arrays.fill(featureMeans, 0);
        float[] featureStDevs = new float[dim];
        for (int dIndex = 0; dIndex < dim; dIndex++) {
            featureStDevs[dIndex] = randa.nextFloat();
        }
        // Set the value bounds.
        float[] lBounds = new float[dim];
        float[] uBounds = new float[dim];
        Arrays.fill(lBounds, -10);
        Arrays.fill(uBounds, 10);
        // Initialize the Gaussian generator.
        MultiDimensionalSphericGaussianGenerator gen =
                new MultiDimens
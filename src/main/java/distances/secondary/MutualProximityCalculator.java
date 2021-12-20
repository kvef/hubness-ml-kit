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
package distances.secondary;

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.Arrays;
import probability.NormalDistributionCalculator;
import sampling.UniformSampler;

/**
 * This classs implements the mutual proximity similarity measure. The basic
 * formula determines the probability that x is a neighbor of y and y is a
 * neighbor of x. This class methods actually return 1 - MP, as distance
 * matrices are the default (and not similarity matrices) throughout the code.
 * The details of the procedure can be found in the following research paper:
 * "Using Mutual Proximity to Improve Content-Based Audio Similarity" that was
 * published in 2011 by Dominik Schnitzer, Arthur Flexer, Markus Schedl and
 * Gerhard Widmer.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MutualProximityCalculator extends CombinedMetric
implements Serializable {

    private static final long serialVersionUID = 1L;
    // Means and standard deviations of distances, for the model.
    double[] distMeans;
    double[] distStDevs;
    // Primary distance matrix.
    float[][] dMatPrimary;
    DataSet dset;
    // CombinedMetric object for distance calculations.
    CombinedMetric cmet;
    
    @Override
    public String toString() {
        return "MutualProximity";
    }

    /**
     * Initialization of the model.
     *
     * @param dMatPrimary Primary distance matrix.
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public MutualProximityCalculator(float[][] dMatPrimary, DataSet dset,
            CombinedMetric cmet) {
        this.dMatPrimary = dMatPrimary;
        this.dset = dset;
        this.cmet = cmet;
        int other;
        if (dMatPrimary != null && dMatPrimary.length > 0) {
            distMeans = new double[dMatPrimary.length];
            distStDevs = new double[dMatPrimary.length];
            float[] numIncluded = new float[dMatPrimary.length];
            // Calculate the means of distances from each point to other points.
            for (int i = 0; i < dMatPrimary.length; i++) {
                for (int j = 0; j < dMatPrimary[i].length; j++) {
                    if (numIncluded[i] == 0) {
                        numIncluded[i] = 1;
                        distMeans[i] = dMatPrimary[i][j];
                    } else {
                        numIncluded[i]++;
                        distMeans[i] = distMeans[i]
                                * ((numIncluded[i] - 1) / numIncluded[i])
                                + dMatPrimary[i][j] * (1 / numIncluded[i]);
                    }
                    other = i + j + 1;
                    if (numIncluded[other] == 0) {
                        numIncluded[other] = 1;
                        distMeans[other] = dMatPrimary[i][j];
                    } else {
                        numIncluded[other]++;
                        distMeans[other] = distMeans[other]
                                * ((numIncluded[other] - 1)
                                / numIncluded[other])
                                + dMatPrimary[i][j] * (1 / numIncluded[other]);
                    }
                }
            }
            Arrays.fill(numIncluded, 0);
            // Calculate the standard deviation of distances from each point to
            // other points.
            for (int i = 0; i < dMatPrimary.length; i++) {
                for (int j = 0; j < dMatPrimary[i].length; j++) {
                    if (numIncluded[i] == 0) {
                        numIncluded[i] = 1;
                        distStDevs[i] = (dMatPrimary[i][j] - distMeans[i])
                                * (dMatPrimary[i][j] - distMeans[i]);
                    } else {
                        numIncluded[i]++;
                        distStDevs[i] = distStDevs[i] * ((numIncluded[i] - 1)
                                / numIncluded[i]) + (dMatPrimary[i][j]
                                - distMeans[i]) * (dMatPrimary[i][j]
                                - distMeans[i]) * (1 / numIncluded[i]);
                    }
                    other = i + j + 1;
                    if (numIncluded[other] == 0) {
                        numIncluded[other] = 1;
                        distStDevs[other] = (dMatPrimary[i][j]
                                - distMeans[other]) * (dMatPrimary[i][j]
                                - distMeans[other]);
                    } else {
                        numIncluded[other]++;
                        distStDevs[other] = distStDevs[other]
                                * ((numIncluded[other] - 1)
                                / numIncluded[other])
                                + (dMatPrimary[i][j] - distMeans[other])
                                * (dMatPrimary[i][j] - distMeans[other])
                                * (1 / numIncluded[other]);
                    }
                }
            }
            for (int i = 0; i < distStDevs.length; i++) {
                distStDevs[i] = Math.sqrt(distStDevs[i]);
            }
        }
    }

    /**
     * Calculate the secondary distance matrix on the data in a multi-threaded
     * way.
     *
     * @param nsf NeighborSetFinder object.
     * @param numThreads Number of threads to use.
     * @return float[][] representing the upper triangular secondary MP distance
     * matrix.
     * @throws Exception
     */
    public float[][] calculateSecondaryDistMatrixMultThr(NeighborSetFinder nsf,
            int numThreads) throws Exception {
        if (dset.isEmpty()) {
            return null;
        } else {
            float[][] distances = new float[dset.size()][];
            int size = dset.size();
            int chunkSize = size / numThreads;
            Thread[] threads = new Thread[numThreads];
            for (int i = 0; i < numThreads - 1; i++) {
                threads[i] = new Thread(new DmCalculator(
                        dset, i * chunkSize, (i + 1) * chunkSize - 1,
                        distances, cmet, nsf));
                threads[i].start();
            }
            threads[numThreads - 1] = new Thread(new DmCalculator(dset,
                    (numThreads - 1) * chunkSize, size - 1,
                    distances, cmet, nsf));
            threads[numThreads - 1].start();
            for (int i = 0; i < numThreads; i++) {
                if (threads[i] != null) {
                    try {
                        threads[i].join();
                    } catch (Throwable t) {
                    }
                }
            }
            return distances;
        }
    }

    /**
     * Calculate the secondary distance matrix on the data in a multi-threaded
     * way, with sampling for speed-up.
     *
     * @param numThreads Integer that is the number of threads to use.
     * @param samplingSize Integer that is the size of the sample.
     * @return float[][] representing the upper triangular secondary MP distance
     * matrix.
     * @throws Exception
     */
    public float[][] calculateSecondaryDistMatrixMultThrFast(
            int numThreads, int samplingSize) throws Exception {
        if (dset.isEmpty()) {
            return null;
        } else {
            samplingSize = Math.min(samplingSize, (int) (dset.size() * 0.8f));
            float[][] distances = new float[dset.size()][];
            int size = dset.size();
            int chunkSize = size / numThreads;
            Thread[] threads = new Thread[numThreads];
            for (int i = 0; i < numThreads - 1; i++) {
                threads[i] = new Thread(new DmCalculatorFast(
                        dset, i * chunkSize, (i + 1) * chunkSize - 1,
                        distances, cmet, samplingSize));
                threads[i].start();
            }
            threads[numThreads - 1] = new Thread(new DmCalculatorFast(dset,
                    (numThreads - 1) * chunkSize, size - 1,
                    distances, cmet, samplingSize));
            threads[numThreads - 1].start();
            for (int i = 0; i < numThreads; i++) {
                if (threads[i] != null) {
                    try {
                        threads[i].join();
                    } catch (Throwable t) {
                    }
                }
            }
            return distances;
        }
    }

    /**
     * Worker class for multi-threaded calculations of the distance matrix.
     */
    class DmCalculator implements Runnable {

        int startRow;
        int endRow;
        float[][] distances;
        CombinedMetric cmet;
        DataSet dset;
        NeighborSetFinder nsf;

        /**
         * The range is inclusive.
         *
         * @param dset DataSet object.
         * @param startRow Index of the start row.
         * @param endRow Index of the end row.
         * @param distances The primary distance matrix.
         * @param cmet The CombinedMetric object used for primary distances.
         * @param nsf The NeighborSetFinder object.
         */
        public DmCalculator(DataSe
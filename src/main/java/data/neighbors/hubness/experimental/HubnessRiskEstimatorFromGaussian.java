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

import data.generators.DataGenerator;
import data.generators.MultiDimensionalSphericGaussianGenerator;
import data.generators.UniformGenerator;
import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import statistics.HigherMoments;
import util.ArrayUtil;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * This class is meant to empirically estimate the distribution of the neighbor
 * occurrence skewness in synthetic data of controlled dimensionality under
 * standard metrics.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubnessRiskEstimatorFromGaussian {

    private File outFile;
    private int k = 5;
    private int kForSecondary = 100;
    private int dataSize = 2000;
    private int dim = 100;
    private int numRepetitions = 500;
    private CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
    private DataGenerator gen;
    private StatsLogger primaryLogger, nicdmLogger, simcosLogger, simhubLogger,
            mpLogger;
    private DataSet dset;
    private float[][] dMatPrimary, dMatSecondary;
    private NeighborSetFinder nsfPrimary, nsfSecondary;
    public static final int NUM_THREADS = 8;
    
    /**
     * This method generates a dataset based on the provided generator.
     * 
     * @return DataSet of synthetic data. 
     */
    private DataSet generateDataSet() {
        if (gen == null) {
            return null;
        }
        DataSet synthSet = new DataSet();
        String[] dummyAttNames = new String[dim];
        for (int d = 0; d < dim; d++) {
            dummyAttNames[d] = "fAtt" + d;
        }
        synthSet.fAttrNames = dummyAttNames;
        synthSet.data = new ArrayList<>(dataSize);
        for (int i = 0; i < dataSize; i++) {
            DataInstance instance = new DataInstance(synthSet);
            instance.fAttr = gen.generateFloat();
            instance.embedInDataset(synthSet);
            synthSet.addDataInstance(instance);
        }
        return synthSet;
    }
    
    /**
     * This method runs the script that examines the risk of hubness in
     * synthetic high-dimensional data.
     */
    private void performAllTests() throws Exception {
        primaryLogger = new StatsLogger("Euclidean");
        nicdmLogger = new StatsLogger("NICDM");
        simcosLogger = new StatsLogger("Simcos");
        simhubLogger = new StatsLogger("Simhub");
        mpLogger = new StatsLogger("MP");
        int kPrimMax = Math.max(k, kForSecondary);
        for (int iteration = 0; iteration < numRepetitions; iteration++) {
            System.out.println("Starting iteration: " + iteration);
            dset = generateDataSet();
            dMatPrimary = dset.calculateDistMatrixMultThr(cmet, NUM_THREADS);
            nsfPrimary = new NeighborSetFinder(dset, dMatPrimary, cmet);
            nsfPrimary.calculateNeighborSets(kPrimMax);
            // We will re-calculate for the smaller k later, now we use this
            // kNN object for secondary distances, where necessary.
            // Calculate the secondary NICDM distances.
            NICDMCalculator nsc = new NICDMCalculator(nsfPrimary);
            dMatSecondary =
                    nsc.getTransformedDMatFromNSFPrimaryDMat();
            nsfSecondary = new NeighborSetFinder(dset, dMatSecondary, nsc);
            nsfSecondary.calculateNeighborSets(k);
            nicdmLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            // Calculate the secondary Simcos distances.
            SharedNeighborFinder snf =
                    new SharedNeighborFinder(nsfPrimary, k);
            snf.setNumClasses(1);
            snf.countSharedNeighborsMultiThread(NUM_THREADS);
            // First fetch the similarities.
            dMatSecondary = snf.getSharedNeighborCounts();
            // Then transform them into distances.
            for (int indexFirst = 0; indexFirst < dMatSecondary.length;
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond <
                        dMatSecondary[indexFirst].length; indexSecond++) {
                    dMatSecondary[indexFirst][ind
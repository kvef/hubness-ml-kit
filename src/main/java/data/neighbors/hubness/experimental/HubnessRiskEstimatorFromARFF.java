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

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import ioformat.FileUtil;
import ioformat.SupervisedLoader;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.methods.knn.HFNN;
import learning.supervised.methods.knn.HIKNN;
import learning.supervised.methods.knn.KNN;
import learning.supervised.methods.knn.NHBNN;
import sampling.UniformSampler;
import statistics.HigherMoments;
import util.ArrayUtil;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * This class is meant to empirically estimate the distribution of the neighbor
 * occurrence skewness in synthetic data of controlled dimensionality under
 * standard metrics. It subsamples a loaded dataset and measures the hubness
 * stats. Additionally, it trains several kNN models and attempts classification
 * on a hold-out sample, measuring its robustness and stability.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HubnessRiskEstimatorFromARFF {
    
    private File outFile;
    private int k = 5;
    private int kForSecondary = 50;
    private int numRepetitions = 500;
    private int sampleSize = 1000;
    private CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
    private StatsLogger primaryLogger, nicdmLogger, simcosLogger, simhubLogger,
            mpLogger;
    private DataSet dsetTrain;
    private DataSet dsetTrainSub;
    private DataSet dsetTest;
    private float[][] dMatPrimaryTrainSub, dMatSecondaryTrainSub;
    private float[][] pointDistancesPrimary, pointDistancesSecondary;
    private int[][] pointNeighborsPrimary, pointNeighborsSecondary,
            pointNeighborsSecondaryK;
    private NeighborSetFinder nsfPrimary, nsfSecondary;
    public static final int NUM_THREADS = 8;
    
    /**
     * This method generates a new data subsample.
     * 
     * @return DataSet that is the subsample of the training data. 
     */
    private DataSet getSample() throws Exception {
        if (dsetTrain == null) {
            return null;
        }
        UniformSampler sampler = new UniformSampler(false);
        DataSet sampleData = sampler.getSample(dsetTrain, sampleSize);
        return sampleData;
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
        KNN knnClassifier;
        NHBNN nhbnnClassifier;
        HIKNN hiknnClassifier;
        HFNN hfnnClassifier;
        float accKNN, accNHBNN, accHIKNN, accHFNN;
        ClassificationEstimator clEstimator;
        int kPrimMax = Math.max(k, kForSecondary);
        DataInstance firstInstance, secondInstance;
        ArrayList<Integer> unitIndexesTest = new ArrayList<>(dsetTest.size());
        for (int i = 0; i < dsetTest.size(); i++) {
            unitIndexesTest.add(i);
        }
        int[] testLabels = dsetTest.obtainLabelArray();
        int numClasses = dsetTest.countCategories();
        for (int iteration = 0; iteration < numRepetitions; iteration++) {
            System.out.println("Starting iteration: " + iteration);
            do {
                dsetTrainSub = getSample();
            } while (dsetTrainSub.countCategories() !=
                    dsetTrain.countCategories());
            dsetTrainSub.orderInstancesByClasses();
            ArrayList<Integer> unitIndexes =
                    new ArrayList<>(dsetTrainSub.size());
            for (int i = 0; i < dsetTrainSub.size(); i++) {
                unitIndexes.add(i);
            }
            dMatPrimaryTrainSub =
                    dsetTrainSub.calculateDistMatrixMultThr(cmet, NUM_THREADS);
            pointDistancesPrimary = new float[dsetTest.size()][
                    dsetTrainSub.size()];
            for (int i = 0; i < dsetTest.size(); i++) {
                for (int j = 0; j < dsetTrainSub.size(); j++) {
                    firstInstance = dsetTest.getInstance(i);
                    secondInstance = dsetTrainSub.getInstance(j);
                    pointDis
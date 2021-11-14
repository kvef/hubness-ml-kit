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
                    pointDistancesPrimary[i][j] = cmet.dist(firstInstance,
                            secondInstance);
                }
            }
            nsfPrimary = new NeighborSetFinder(dsetTrainSub,
                    dMatPrimaryTrainSub, cmet);
            nsfPrimary.calculateNeighborSets(kPrimMax);
            // We will re-calculate for the smaller k later, now we use this
            // kNN object for secondary distances, where necessary.
            // Calculate the secondary NICDM distances.
            NICDMCalculator nsc = new NICDMCalculator(nsfPrimary);
            dMatSecondaryTrainSub =
                    nsc.getTransformedDMatFromNSFPrimaryDMat();
            nsfSecondary = new NeighborSetFinder(dsetTrainSub,
                    dMatSecondaryTrainSub, nsc);
            nsfSecondary.calculateNeighborSets(k);
            nicdmLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            nicdmLogger.updateLabelMismatchPercentages(
                    nsfSecondary.getKNeighbors());
            pointDistancesSecondary = new float[dsetTest.size()][
                    dsetTrainSub.size()];
            pointNeighborsSecondary = new int[dsetTest.size()][k];
            pointNeighborsSecondaryK = new int[dsetTest.size()][kForSecondary];
            for (int index = 0; index < dsetTest.size(); index++) {
                firstInstance = dsetTest.getInstance(index);
                pointNeighborsSecondaryK[index] =
                        NeighborSetFinder.getIndexesOfNeighbors(
                        dsetTrainSub, firstInstance, kForSecondary,
                        pointDistancesPrimary[index]);
            }
            for (int indexFirst = 0; indexFirst < dsetTest.size();
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond < dsetTrainSub.size();
                        indexSecond++) {
                    firstInstance = dsetTest.getInstance(indexFirst);
                    secondInstance = dsetTrainSub.getInstance(indexSecond);
                    int[] firstNeighbors = pointNeighborsSecondaryK[indexFirst];
                    float[] kDistsFirst = new float[kForSecondary];
                    float[] kDistsSecond = nsfPrimary.getKDistances()[
                            indexSecond];
                    for (int kInd = 0; kInd < kForSecondary; kInd++) {
                        kDistsFirst[kInd] = pointDistancesPrimary[indexFirst][
                                firstNeighbors[kInd]];
                    }
                    pointDistancesSecondary[indexFirst][indexSecond] =
                            nsc.distFromKDists(firstInstance, secondInstance,
                            kDistsFirst, kDistsSecond);
                }
            }
            pointNeighborsSecondary =
                    NeighborSetFinder.getIndexesOfNeighbors(dsetTrainSub,
                    dsetTest, k, pointDistancesSecondary);
            // Initialize the classifiers.
            knnClassifier = new KNN(k, nsc);
            nhbnnClassifier = new NHBNN(k, nsc, numClasses);
            hiknnClassifier = new HIKNN(k, nsc, numClasses);
            hfnnClassifier = new HFNN(k, nsc, numClasses);
            // Set the data.
            knnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            nhbnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hiknnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hfnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            // Set the distances and the kNN sets.
            nhbnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            nhbnnClassifier.setNSF(nsfSecondary);
            hiknnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hiknnClassifier.setNSF(nsfSecondary);
            hfnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hfnnClassifier.setNSF(nsfSecondary);
            // Train the models.
            knnClassifier.train();
            nhbnnClassifier.train();
            hiknnClassifier.train();
            hfnnClassifier.train();
            // Test the classifiers.
            clEstimator = knnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accKNN = clEstimator.getAccuracy();
            clEstimator = nhbnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accNHBNN = clEstimator.getAccuracy();
            clEstimator = hiknnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHIKNN = clEstimator.getAccuracy();
            clEstimator = hfnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHFNN = clEstimator.getAccuracy();
            nicdmLogger.updateByClassifierAccuracies(accKNN, accNHBNN, accHIKNN,
                    accHFNN);
            // Calculate the secondary Simcos distances.
            SharedNeighborFinder snf =
                    new SharedNeighborFinder(nsfPrimary, k);
            snf.setNumClasses(numClasses);
            snf.countSharedNeighborsMultiThread(NUM_THREADS);
            // First fetch the similarities.
            dMatSecondaryTrainSub = snf.getSharedNeighborCounts();
            // Then transform them into distances.
            for (int indexFirst = 0; indexFirst < dMatSecondaryTrainSub.length;
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond <
                        dMatSecondaryTrainSub[indexFirst].length;
                        indexSecond++) {
                    dMatSecondaryTrainSub[indexFirst][indexSecond] =
                            kForSecondary -
                            dMatSecondaryTrainSub[indexFirst][indexSecond];
                }
            }
            SharedNeighborCalculator snc =
                    new SharedNeighborCalculator(snf,SharedNeighborCalculator.
                    WeightingType.NONE);
            nsfSecondary = new NeighborSetFinder(dsetTrainSub,
                    dMatSecondaryTrainSub, snc);
            nsfSecondary.calculateNeighborSets(k);
            simcosLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            simcosLogger.updateLabelMismatchPercentages(
                    nsfSecondary.getKNeighbors());
            // Calculate the test-to-training point distances.
            pointDistancesSecondary = new float[dsetTest.size()][
                    dsetTrainSub.size()];
            pointNeighborsSecondary = new int[dsetTest.size()][k];
            pointNeighborsSecondaryK = new int[dsetTest.size()][kForSecondary];
            for (int index = 0; index < dsetTest.size(); index++) {
                firstInstance = dsetTest.getInstance(index);
                pointNeighborsSecondaryK[index] =
                        NeighborSetFinder.getIndexesOfNeighbors(
                        dsetTrainSub, firstInstance, kForSecondary,
                        pointDistancesPrimary[index]);
            }
            for (int indexFirst = 0; indexFirst < dsetTest.size();
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond < dsetTrainSub.size();
                        indexSecond++) {
                    firstInstance = dsetTest.getInstance(indexFirst);
                    secondInstance = dsetTrainSub.getInstance(indexSecond);
                    pointDistancesSecondary[indexFirst][indexSecond] =
                            snc.dist(firstInstance, secondInstance,
                            pointNeighborsSecondaryK[indexFirst],
                            nsfPrimary.getKNeighbors()[indexSecond]);
                }
            }
            pointNeighborsSecondary =
                    NeighborSetFinder.getIndexesOfNeighbors(dsetTrainSub,
                    dsetTest, k, pointDistancesSecondary);
            // Initialize the classifiers.
            knnClassifier = new KNN(k, snc);
            nhbnnClassifier = new NHBNN(k, snc, numClasses);
            hiknnClassifier = new HIKNN(k, snc, numClasses);
            hfnnClassifier = new HFNN(k, snc, numClasses);
            // Set the data.
            knnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            nhbnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hiknnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hfnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            // Set the distances and the kNN sets.
            nhbnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            nhbnnClassifier.setNSF(nsfSecondary);
            hiknnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hiknnClassifier.setNSF(nsfSecondary);
            hfnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hfnnClassifier.setNSF(nsfSecondary);
            // Train the models.
            knnClassifier.train();
            nhbnnClassifier.train();
            hiknnClassifier.train();
            hfnnClassifier.train();
            // Test the classifiers.
            clEstimator = knnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accKNN = clEstimator.getAccuracy();
            clEstimator = nhbnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accNHBNN = clEstimator.getAccuracy();
            clEstimator = hiknnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHIKNN = clEstimator.getAccuracy();
            clEstimator = hfnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHFNN = clEstimator.getAccuracy();
            simcosLogger.updateByClassifierAccuracies(accKNN, accNHBNN,
                    accHIKNN, accHFNN);
            // Calculate the secondary Simhub distances. These are actually the
            // simhub^inf variant, since there are not classes in the data.
            snf = new SharedNeighborFinder(nsfPrimary, k);
            snf.setNumClasses(numClasses);
            snf.obtainWeightsFromHubnessInformation();
            snf.countSharedNeighborsMultiThread(NUM_THREADS);
            // First fetch the similarities.
            dMatSecondaryTrainSub = snf.getSharedNeighborCounts();
            // Then transform them into distances.
            for (int indexFirst = 0; indexFirst < dMatSecondaryTrainSub.length;
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond <
                        dMatSecondaryTrainSub[indexFirst].length;
                        indexSecond++) {
                    dMatSecondaryTrainSub[indexFirst][indexSecond] =
                            kForSecondary -
                            dMatSecondaryTrainSub[indexFirst][indexSecond];
                }
            }
            snc = new SharedNeighborCalculator(snf,SharedNeighborCalculator.
                    WeightingType.HUBNESS_INFORMATION);
            nsfSecondary = new NeighborSetFinder(dsetTrainSub,
                    dMatSecondaryTrainSub, snc);
            nsfSecondary.calculateNeighborSets(k);
            simhubLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            simhubLogger.updateLabelMismatchPercentages(
                    nsfSecondary.getKNeighbors());
            pointDistancesSecondary = new float[dsetTest.size()][
                    dsetTrainSub.size()];
            pointNeighborsSecondary = new int[dsetTest.size()][k];
            pointNeighborsSecondaryK = new int[dsetTest.size()][kForSecondary];
            for (int index = 0; index < dsetTest.size(); index++) {
                firstInstance = dsetTest.getInstance(index);
                pointNeighborsSecondaryK[index] =
                        NeighborSetFinder.getIndexesOfNeighbors(
                        dsetTrainSub, firstInstance, kForSecondary,
                        pointDistancesPrimary[index]);
            }
            for (int indexFirst = 0; indexFirst < dsetTest.size();
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond < dsetTrainSub.size();
                        indexSecond++) {
                    firstInstance = dsetTest.getInstance(indexFirst);
                    secondInstance = dsetTrainSub.getInstance(indexSecond);
                    pointDistancesSecondary[indexFirst][indexSecond] =
                            snc.dist(firstInstance, secondInstance,
                            pointNeighborsSecondaryK[indexFirst],
                            nsfPrimary.getKNeighbors()[indexSecond]);
                }
            }
            pointNeighborsSecondary =
                    NeighborSetFinder.getIndexesOfNeighbors(dsetTrainSub,
                    dsetTest, k, pointDistancesSecondary);
            // Initialize the classifiers.
            knnClassifier = new KNN(k, snc);
            nhbnnClassifier = new NHBNN(k, snc, numClasses);
            hiknnClassifier = new HIKNN(k, snc, numClasses);
            hfnnClassifier = new HFNN(k, snc, numClasses);
            // Set the data.
            knnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            nhbnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hiknnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            hfnnClassifier.setDataIndexes(unitIndexes, dsetTrainSub);
            // Set the distances and the kNN sets.
            nhbnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            nhbnnClassifier.setNSF(nsfSecondary);
            hiknnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hiknnClassifier.setNSF(nsfSecondary);
            hfnnClassifier.setDistMatrix(dMatSecondaryTrainSub);
            hfnnClassifier.setNSF(nsfSecondary);
            // Train the models.
            knnClassifier.train();
            nhbnnClassifier.train();
            hiknnClassifier.train();
            hfnnClassifier.train();
            // Test the classifiers.
            clEstimator = knnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accKNN = clEstimator.getAccuracy();
            clEstimator = nhbnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accNHBNN = clEstimator.getAccuracy();
            clEstimator = hiknnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHIKNN = clEstimator.getAccuracy();
            clEstimator = hfnnClassifier.test(unitIndexesTest, dsetTest,
                    testLabels, numClasses, pointDistancesSecondary,
                    pointNeighborsSecondary);
            accHFNN = clEstimator.getAccuracy();
            simhubLogger.updateByClassifierAccuracies(accKNN, accNHBNN,
                    accHIKNN, accHFNN);
            // Calculate the secondary Mutual Proximity distances.
            MutualProximityCalculator calc =
                    new MutualProximityCalculator(nsfPrimary.getDistances(),
                    nsfPrimary.getDataSet(), nsfPrimary.getCombinedMetric());
            dMatSecondaryTrainSub = calc.calculateSecondaryDistMatrixMultThr(
                    nsfPrimary, 8);
            nsfSecondary = new NeighborSetFinder(dsetTrainSub,
                    dMatSecondaryTrainSub, calc);
            nsfSecondary.calculateNeighborSets(k);
            mpLogger.updateByObservedFreqs(
                    nsfSecondary.getNeighborFrequencies());
            mpLogger.updateLabelMismatchPercentages(
                    nsfSecondary.getKNeighbors());
            pointDistancesSecondary = new float[dsetTest.size()][
                    dsetTrainSub.size()];
            pointNeighborsSecondary = new int[dsetTest.size()][k];
            pointNeighborsSecondaryK = new int[dsetTest.size()][kForSecondary];
            for (int index = 0; index < dsetTest.size(); index++) {
                firstInstance = dsetTest.getInstance(index);
                pointNeighborsSecondaryK[index] =
                        NeighborSetFinder.getIndexesOfNeighbors(
                        dsetTrainSub, firstInstance, kForSecondary,
                        pointDistancesPrimary[index]);
            }
            for (int indexFirst = 0; indexFirst < dsetTest.size();
                    indexFirst++) {
                for (int indexSecond = 0; indexSecond < dsetTrainSub.size();
                        indexSecond++) {
                    firstInstance = dsetTest.getInstance(indexFirst);
                    secondInstance = dsetTrainSub.getInstance(indexSecond);
                    int[] firstNeighbors = pointNeighborsSecondaryK[
                            indexFirst];
                    float[] kDistsFirst = new float[kForSecondary];
                    float[] kDistsSecond =
                            nsfPrimary.getKDistances()[indexSecond];
                    for (int kInd = 0; kInd < kForSecondary; kInd++) {
                        kDistsFirst[kInd] = pointDistancesPrimary[indexFirst][
                                firstNeighbors[kInd]];
                    }
                    pointDistancesSecondary[indexFirst][indexSecond] =
                            calc.dist(firstInstance, secondInstance,
                            kDistsFirst, kDistsSecond);
                }
            }
            pointNeighborsSecondary =
                    NeighborSetFinder.getIndexesOfNeighbors(dsetTrainSub,
                    dsetTest, k, pointDistancesSecondary);
            // Initialize the classifiers.
           
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
package data.neighbors.hubness;

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import distances.primary.DistanceMeasure;
import distances.secondary.LocalScalingCalculator;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import distances.sparse.SparseCombinedMetric;
import distances.sparse.SparseMetric;
import filters.TFIDF;
import ioformat.DistanceMatrixIO;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import ioformat.SupervisedLoader;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import learning.supervised.evaluation.cv.BatchClassifierTester;
import util.BasicMathUtil;

/**
 * This class acts as a script for batch analysis of hubness stats on a series
 * of datasets. It is meant for the multi-label case, when several different
 * label arrays are provided for the datasets. Different datasets are different
 * feature representations of the same underlying data. This is different from
 * the BatchHubnessAnalyzer class, which handles the general batch processing
 * case. However, this class is more efficient for multi-label analysis, as the
 * distance matrices and kNN sets are only calculated once for each data
 * representation.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MultiLabelBatchHubnessAnalyzer {

    // Normalization types.
    private static final int NORM_STANDARDIZE = 0;
    private static final int NORM_NO = 1;
    private static final int NORM_01 = 2;
    private static final int N_TFIDF = 3;
    // Normalization to use on the features.
    private int normType = NORM_STANDARDIZE;
    private BatchClassifierTester.SecondaryDistance secondaryDistanceType;
    // Neighborhood size to use for secondary distances.
    private int secondaryDistanceK = 50;
    // The upper bound on the neighborhood size to test. All smaller 
    // neighborhood sizes will be examined.
    private int kMax = 50;
    // Noise and mislabeling range definitions, with default values.
    private float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep = 1;
    // Input and output files and directories.
    private File inConfigFile, inDir, outDir, currOutDSDir, inLabelFile;
    // Dataset paths.
    private ArrayList<String> dsPaths = new ArrayList<>(100);
    // A list of corresponding dataset metrics.
    private ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    // Data holders.
    private DataSet originalDSet, currDSet;
    // The current metric object.
    private CombinedMetric cmet;
    // Number of categories in the data.
    private int numCategories;
    // Label separator in the label file.
    private String labelSeparator;
    // Directory containing the distances.
    File distancesDir;

    /**
     * Initialization.
     *
     * @param inConfigFile File containing the experiment configuration.
     */
    public MultiLabelBatchHubnessAnalyzer(File inConfigFile) {
        this.inConfigFile = inConfigFile;
    }

    /**
     * This method runs the script and performs batch analysis of the stats
     * relevant for interpreting the hubness of the data on a series on
     * datasets.
     *
     * @throws Exception
     */
    public void runAllTests() throws Exception {
        // First load different data label arrays.
        DataSet labelDataset = null;
        int labelArrayLength;
        if (inLabelFile.getPath().endsWith(".arff")) {
            // Each label array is a column, i.e. corresponds to a feature in
            // the dataset.
            IOARFF aPers = new IOARFF();
            labelDataset = aPers.load(inLabelFile.getPath());
        } else if (inLabelFile.getPath().endsWith(".csv")) {
            IOCSV reader = new IOCSV(false, labelSeparator,
                    DataMineConstants.INTEGER);
            labelDataset = reader.readData(inLabelFile);
        } else {
            System.out.println("Wrong label format.");
            throw new Exception();
        }
        labelArrayLength = labelDataset.getNumIntAttr();
        int dsIndex = 0;
        for (String dsPath : dsPaths) {
            File dsFile = new File(dsPath);
            // Load in the multi-label mode.
            originalDSet = SupervisedLoader.loadData(dsFile, true);
            System.out.println("Testing on: " + dsPath);
            originalDSet.standardizeCategories();
            // Count the categories in the data.
            numCategories = originalDSet.countCategories();
            if (normType != NORM_NO) {
                System.out.print("Normalizing features-");
                if (normType == NORM_01) {
                    // Normalize all float features to the [0, 1] range.
                    originalDSet.normalizeFloats();
                } else if (normType == NORM_STANDARDIZE) {
                    // Standardize all float values.
                    originalDSet.standardizeAllFloats();
                } else if (normType == N_TFIDF) {
                    // Perform TFIDF weighting.
                    boolean[] fBool;
                    if (originalDSet instanceof BOWDataSet) {
                        fBool = new boolean[((BOWDataSet) originalDSet).
                                getNumDifferentWords()];
                    } else {
                        fBool = new boolean[originalDSet.getNumFloatAttr()];
                    }
                    Arrays.fill(fBool, true);
                    TFIDF filterTFIDF = new TFIDF(fBool,
                            DataMineConstants.FLOAT);
                    if (originalDSet instanceof BOWDataSet) {
                        filterTFIDF.setSparse(true);
                    }
                    filterTFIDF.filter(originalDSet);
                }
                System.out.println("-Normalization complete.");
            } else {
                System.out.println("Skipping feature normalization.");
            }
            // First iterate over different noise levels.
            for (float noise = noiseMin; noise <= noiseMax; noise +=
                    noiseStep) {
                System.gc();
                currDSet = originalDSet.copy();
                // Add noise if a positive noise level was indicated.
                if (noise > 0) {
                    currDSet.addGaussianNoiseToNormalizedCollection(
                            noise, 0.1f);
                }
                for (int lIndex = 0; lIndex < labelArrayLength; lIndex++) {
                    // Assign labels to the data.
                    // Iterate over the mislabeling levels.
                    for (float ml = mlMin; ml <= mlMax; ml += mlStep) {
                        for (int dInd = 0; dInd < currDSet.size(); dInd++) {
                            currDSet.data.get(dInd).setCategory(
                                    labelDataset.data.get(dInd).iAttr[lIndex]);
                        }
                        // Induce mislabeling, if specified.
                        if (ml > 0) {
                            currDSet.induceMislabeling(ml, numCategories);
                        }
                        // Calculate the out directory.
                        currOutDSDir = new File(outDir,
                                dsFile.getName().substring(0,
                                dsFile.getName().lastIndexOf(".")) + "l"
                                + lIndex + File.separator + "k" + kMax
                                + File.separator + "ml" + ml + File.separator
                                + "noise" + noise);
                        FileUtil.createDirectory(currOutDSDir);
                        // Get the appropriate metric.
                        cmet = dsMetric.get(dsIndex);
                        // Calculate class priors.
                        float[] classPriors = currDSet.getClassPriors();
                        // Calculate the k-nearest neighbor sets.
                        NeighborSetFinder nsf = new NeighborSetFinder(
                                currDSet, cmet);
                        // Determine the correct distance matrix path.
                        String dMatPath = null;
                        if (distancesDir != null) {
                            if (!(cmet instanceof SparseCombinedMetric)) {
                                switch (normType) {
                                    case NORM_NO:
                                        dMatPath = "NO" + File.separator
                                                + cmet.getFloatMetric().
                                                getClass().getName() +
                                                File.separator + "dMat.txt";
                                        break;
                                    case NORM_01:
                                        dMatPath = "NORM01" + File.separator
                                                + cmet.getFloatMetric().
                                                getClass().getName() +
                                                File.separator + "dMat.txt";
                                        break;
                                    case NORM_STANDARDIZE:
                                        dMatPath = "STANDARDIZED" +
                                                File.separator + cmet.
                                                getFloatMetric().getClass().
                                                getName() + File.separator
                                                + "dMat.txt";
                                        break;
                                    case N_TFIDF:
                                        dMatPath = "TFIDF" + File.separator
                                                + cmet.getFloatMetric().
                                                getClass().getName() +
                                                File.separator + "dMat.txt";
                                        break;
                                }
                            } else {
                                switch (normType) {
                                    case NORM_NO:
                                        dMatPath = "NO" + File.separator
                                                + ((SparseCombinedMetric) cmet).
                                                getSparseMetric().getClass().
                                                getName() + File.separator
                                                + "dMat.txt";
                                        break;
                                    case NORM_01:
                                        dMatPath = "NORM01" + File.separator
                                                + ((SparseCombinedMetric) cmet).
                                                getSparseMetric().getClass().
                                                getName() + File.separator
                                                + "dMat.txt";
                                        break;
                                    case NORM_STANDARDIZE:
                                        dMatPath = "STANDARDIZED" +
                                                File.separator +
                                                ((SparseCombinedMetric) cmet).
                                                getSparseMetric().getClass().
                                                getName() + File.separator
                                                + "dMat.txt";
                                        break;
                                    case N_TFIDF:
                                        dMatPath = "TFIDF" + File.separator
                                                + ((SparseCombinedMetric) cmet).
                                                getSparseMetric().getClass().
                                                getName() + File.separator
                                                + "dMat.txt";
                                        break;
                                }
                            }
                        }
                        File dMatFile = null;
                        float[][] distMat = null;
                        // First initialize with a dummy class object, to avoid
                        // some warnings and exceptions in the pathological
                        // cases.
                        Class cmetClass = originalDSet.getClass();
                        // Determine the proper metric class and the distance
                        // matrix file.
                        if (dMatPath != null && noise == 0) {
                            dMatFile = new File(
                                    distancesDir, dsFile.getName().substring(0,
                                    dsFile.getName().lastIndexOf("."))
                                    + File.separator + dMatPath);
                            cmetClass = Class.forName(
                                    dMatFile.getParentFile().getName());
                        }
                        if (distMat == null) {
                            if (dMatFile == null
                                    || !dMatFile.exists()
                                    || !(cmetClass.isInstance(
                                    cmet.getFloatMetric()))) {
                                // If the file does not exist or the loaded name
                                // is not an appropriate float metric, then
                                // calculate the distances with the specified
                                // metric.
                                System.out.print("Calculating distances-");
                                distMat = currDSet.calculateDistMatrixMultThr(
                                        cmet, 4);
                                System.out.println("-distances calculated.");
                                if (dMatFile != null) {
                                    // If the file path is good, persist the
                                    // newly calculated distance matrix.
                                    DistanceMatrixIO.printDMatToFile(
                                            distMat, dMatFile);
                                }
                            } else {
                                // Load the distances from an existing source.
                                System.out.print("Loading distances-");
                                distMat = DistanceMatrixIO.loadDMatFromFile(
                                        dMatFile);
                                System.out.println("-distance loaded from "
                                        + "file: " + dMatFile.getPath());
                            }
                        }
                        if (secondaryDistanceType == BatchClassifierTester.
                                SecondaryDistance.NONE) {
                            // Use the primary distance matrix for kNN
                            // calculations.
                            nsf.setDistances(distMat);
                        } else {
                            // Use the secondary shared-neighbor distances.
                            if (secondaryDistanceType == BatchClassifierTester.
                                    SecondaryDistance.SIMCOS) {
                                // The simcos secondary distance.
                                NeighborSetFinder nsfSND =
                                        new NeighborSetFinder(currDSet, distMat,
                                        cmet);
                                nsfSND.calculateNeighborSetsMultiThr(
                                        secondaryDistanceK, 8);
                                SharedNeighborFinder snf =
                                        new SharedNeighborFinder(nsfSND);
                                snf.countSharedNeighbors();
                                SharedNeighborCalculator snc =
                                        new SharedNeighborCalculator(snf,
                                        SharedNeighborCalculator.
                                        WeightingType.NONE);
                                float[][] simcosSimMat =
                                        snf.getSharedNeighborCounts();
                                float[][] simcosDMat =
                                        new float[simcosSimMat.length][];
                                // Transform similarities into distances.
                                for (int i = 0; i < simcosDMat.length; i++) {
                                    simcosDMat[i] =
                                            new float[simcosSimMat[i].length];
                                    for (int j = 0;
                                            j < simcosDMat[i].length; j++) {
                                        simcosDMat[i][j] = secondaryDistanceK
                                                - simcosSimMat[i][j];
                                    }
                                }
                                // Normalize the scores.
                                float max = 0;
                                float min = Float.MAX_VALUE;
                                for (int i = 0; i < simcosDMat.length; i++) {
                                    for (int j = 0; j < simcosDMat[i].length;
                                            j++) {
                                        max = Math.max(max, simcosDMat[i][j]);
                                        min = Math.min(min, simcosDMat[i][j]);
                                    }
                                }
                                for (int i = 0; i < simcosDMat.length; i++) {
                                    for (int j = 0; j < simcosDMat[i].length;
                                            j++) {
                                        simcosDMat[i][j] =
                                                (simcosDMat[i][j] - min)
                                                / (max - min);
                                    }
                                }
                                // Use the simcos distance matrix for kNN set
                                // calculations.
                                nsf = new NeighborSetFinder(originalDSet,
                                        simcosDMat, snc);
                            } else if (secondaryDistanceType
                                    == BatchClassifierTester.
                                    SecondaryDistance.SIMHUB) {
                                // The hubness-aware simhub secondary distance
                                // measure based on shared-neighbor methodology.
                                NeighborSetFinder nsfSND =
                                        new NeighborSetFinder(
                                        currDSet, distMat, cmet);
                                nsfSND.calculateNeighborSetsMultiThr(
                                        secondaryDistanceK, 8);
                                SharedNeighborFinder snf =
                                        new SharedNeighborFinder(nsfSND, 5);
                                snf.obtainWeightsFromHubnessInformation(0);
                                snf.countSharedNeighbors();
                                SharedNeighborCalculator snc =
                                        new SharedNeighborCalculator(snf,
                                        Shar
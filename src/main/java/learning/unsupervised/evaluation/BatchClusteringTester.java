
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
package learning.unsupervised.evaluation;

import configuration.BatchClusteringConfig;
import data.neighbors.NeighborSetFinder;
import data.neighbors.approximate.AppKNNGraphLanczosBisection;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.discrete.tranform.EntropyMDLDiscretizer;
import data.representation.sparse.BOWDataSet;
import data.representation.util.DataMineConstants;
import distances.kernel.Kernel;
import distances.kernel.KernelMatrixUserInterface;
import distances.primary.CombinedMetric;
import distances.sparse.SparseCombinedMetric;
import feature.evaluation.Info;
import filters.TFIDF;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.ArrayList;
import learning.supervised.interfaces.DistMatrixUserInterface;
import data.neighbors.NSFUserInterface;
import data.neighbors.SharedNeighborFinder;
import distances.secondary.LocalScalingCalculator;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import ioformat.DistanceMatrixIO;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import learning.supervised.evaluation.cv.BatchClassifierTester.SecondaryDistance;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClustererFactory;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.evaluation.quality.QIndexDunn;
import learning.unsupervised.evaluation.quality.QIndexIsolation;
import learning.unsupervised.evaluation.quality.QIndexRand;
import learning.unsupervised.evaluation.quality.QIndexSilhouette;
import learning.unsupervised.methods.DBScan;

import probability.GaussianMixtureModel;
import probability.Perplexity;
import sampling.UniformSampler;

/**
 * This class implements the functionality for cross-algorithm clustering
 * comparisons on multiple datasets with multiple metrics, under possible
 * inclusion of feature noiseRate or on the exact feature vectors. Acceptable
 * data formats include ARFF and CSV. There is an option of splitting the data
 * to training and test sets automatically for evaluation as well - though it is
 * not strictly speaking necessary for avoiding over-fitting in clustering.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchClusteringTester {

    // Whether to calculate model perplexity as a quality measure, as it can be
    // very slow for larger and high-dimensional datasets, prohibitively so.
    private boolean calculatePerplexity = false;
    // Whether to use split testing (training/test data).
    private boolean splitTesting = false;
    private float splitPerc = 1;
    // Whether a specific number of desired clusters was specified.
    private boolean nClustSpecified = true;
    private volatile int globalIterationCounter;
    // Applicable types of feature normalization.

    public enum Normalization {

        NONE, STANDARDIZE, NORM_01, TFIDF;
    }
    // The normalization type to actually use in the experiments.
    private Normalization normType = Normalization.STANDARDIZE;
    private boolean clustersAutoSet = false;
    private double execTimeAllOneRun; // In miliseconds.
    // The number of times a clustering is performed on every single dataset.
    private int timesOnDataSet = 30;
    // The preferred number of iterations, where applicable.
    private int minIter;
    private ArrayList<String> clustererNames = new ArrayList<>(10);
    // Possible parameter value maps to use with certain algorithms.
    public HashMap<String, HashMap<String, Object>> algorithmParametrizationMap;
    // The experimental neighborhood range, with default values.
    private int kMin = 5, kMax = 5, kStep = 1;
    private int kGenMin = 1;
    // Noise and mislabeling experimental ranges, with default values.
    private float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep = 1;
    // Files pointing to the input and output.
    private File inConfigFile, inDir, outDir, distancesDir, mlWeightsDir;
    // Paths to the tested datasets.
    private ArrayList<String> dsPaths = new ArrayList<>(100);
    // Paths to the corresponding metrics for the datasets.
    private ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    private DataSet originalDSet, currDSet;
    private DiscretizedDataSet currDiscDSet;
    private CombinedMetric currCmet;
    private int[] originalLabels;
    private int[] trainingLabels;
    private int[] testLabels;
    private int numCategories;
    private boolean[] discrete;
    private float[][] distMat;
    private float[][] kMat;
    private float[][] kernelDMat;
    private Kernel ker = null;
    // Interface user flags.
    private boolean distUserPresentNonDisc = false;
    private boolean distUserPresentDisc = false;
    private boolean kernelUserPresent = false;
    private boolean discreteExists;
    private boolean nsfUserPresent;
    private boolean kernelNSFUserPresent;
    // Secondary distance specification.
    private SecondaryDistance secondaryDistanceType;
    private int secondaryK = 50;
    // Clustering range.
    private int cluNumMin;
    private int cluNumMax;
    private int cluNumStep;
    // Aproximate kNN calculations specification.
    private float approximateNeighborsAlpha = 1f;
    private boolean approximateNeighbors = false;
    // The number of threads used for distance matrix and kNN set calculations.
    private int numCommonThreads = 8;

    /**
     * Reads the parameters from the configuration file.
     *
     * @param inConfigFile Configuration file containing all the parameters. The
     * exact format of the configuration file can be discerned from the
     * loadParameters() method in this class.
     */
    public BatchClusteringTester(File inConfigFile) {
        this.inConfigFile = inConfigFile;
    }

    public void runAllTests() throws Exception {
        // Index of the currently examined dataset.
        int dsCounter = 0;
        // We iterate over dataset paths.
        for (String dsPath : dsPaths) {
            // This loads the appropriate metrics object for this particular
            // dataset.
            currCmet = dsMetric.get(dsCounter);
            File dsFile = new File(dsPath);
            // It is possible to indicate that the data in the file is given in
            // sparse format by precluding the dataset name with "sparse:". This
            // code chops off the prefix and extracts the true data name and
            // loads the data in.
            if (dsPath.startsWith("sparse:")) {
                String trueDSPath = dsPath.substring(
                        dsPath.indexOf(':') + 1, dsPath.length());
                IOARFF pers = new IOARFF();
                originalDSet = pers.loadSparse(trueDSPath);
            } else {
                // This is for usual, dense data representations.
                if (dsPath.endsWith(".csv")) {
                    try {
                        // First we try reading it as if the data were
                        // comma-separated.
                        IOCSV reader = new IOCSV(true, ",");
                        originalDSet = reader.readData(dsFile);
                    } catch (Exception e) {
                        try {
                            // If not comma, then empty spaces.
                            IOCSV reader = new IOCSV(true, " +");
                            originalDSet = reader.readData(dsFile);
                        } catch (Exception e1) {
                            // Prior attempts have all assumed there was a
                            // category label as the last attribute. This
                            // one doesn't - the data is loaded without the
                            // labels.
                            try {
                                IOCSV reader = new IOCSV(false, ",");
                                originalDSet = reader.readData(dsFile);
                            } catch (Exception e2) {
                                IOCSV reader = new IOCSV(false, " +");
                                originalDSet = reader.readData(dsFile);
                            }
                        }
                    }
                } else if (dsPath.endsWith(".tsv")) {
                    // Similar as above, but now for .tsv files instead of .csv
                    try {
                        IOCSV reader = new IOCSV(true, " +");
                        originalDSet = reader.readData(dsFile);
                    } catch (Exception e) {
                        try {
                            IOCSV reader = new IOCSV(true, "\t");
                            originalDSet = reader.readData(dsFile);
                        } catch (Exception e1) {
                            try {
                                IOCSV reader = new IOCSV(false, " +");
                                originalDSet =
                                        reader.readData(dsFile);
                            } catch (Exception e2) {
                                IOCSV reader = new IOCSV(false, "\t");
                                originalDSet =
                                        reader.readData(dsFile);
                            }
                        }
                    }
                } else if (dsPath.endsWith(".arff")) {
                    // Similar as above, though now for .arff files.
                    IOARFF persister = new IOARFF();
                    originalDSet = persister.load(dsPath);
                } else {
                    // If everything fails, report an error.
                    System.out.println("Error, could not read: " + dsPath);
                    continue;
                }
            }
            System.out.println(" Testing on: " + dsPath);
            // Category standardization ensures the class labels are subsequent
            // integers, i.e. there are no 'holes' like 1,2,4,6,7. This would
            // have been standardized to 1,2,3,4,5.
            originalDSet.standardizeCategories();
            if (!nClustSpecified) {
                // This testing mode is for algorithms that can determine the
                // optimal cluster number on their own.
                clustersAutoSet = true;
            }
            if (clustersAutoSet) {
                // Those algorithms that do not determine the optimal number of
                // clusters on their own have the predefined cluster number set
                // to the number of categories in the data. The number of
                // clusters is set to 2 for those datasets that have no labels.
                int numCat = originalDSet.countCategories();
                cluNumMin = Math.max(numCat, 2);
                cluNumMax = Math.max(numCat, 2);
                cluNumStep = 1;
            }
            System.out.print(" Normalizing features-");
            // Apply the chosen feature normalization.
            if (normType == Normalization.NORM_01) {
                originalDSet.normalizeFloats();
            } else if (normType == Normalization.STANDARDIZE) {
                originalDSet.standardizeAllFloats();
            } else if (normType == Normalization.TFIDF) {
                boolean[] fBool;
                if (originalDSet instanceof BOWDataSet) {
                    // The sparse case.
                    fBool = new boolean[((BOWDataSet) originalDSet).
                            getNumDifferentWords()];
                } else {
                    // The dense case.
                    fBool = new boolean[originalDSet.getNumFloatAttr()];
                }
                Arrays.fill(fBool, true);
                TFIDF filterTFIDF = new TFIDF(fBool, DataMineConstants.FLOAT);
                if (originalDSet instanceof BOWDataSet) {
                    filterTFIDF.setSparse(true);
                }
                filterTFIDF.filter(originalDSet);
            }
            System.out.println("-normalization complete");
            // The original labels are stored in a separate arrau.
            originalLabels = originalDSet.obtainLabelArray();
            trainingLabels = originalLabels;
            numCategories = originalDSet.countCategories();
            for (int i = 0; i < clustererNames.size(); i++) {
                String cName = clustererNames.get(i);
                if (cName.equals("dbscan")) {
                    // DBScan requires a certain neighborhood size - so even
                    // if smaller max neighborhood sizes are requrested
                    // explicitly, implicitly a larger one might actually
                    // be calculated in the embedding NeighborSetFinder
                    // object - if DBScan is among the tested approaches.
                    nsfUserPresent = true;
                    kGenMin = 20;
                }
                if (isDiscrete(cName)) {
                    // Works on discretized representations.
                    discreteExists = true;
                }
                if (requiresNSF(cName)) {
                    // Requires neighbor sets to be provided.
                    nsfUserPresent = true;
                }
                if (requiresKernelNSF(cName)) {
                    // Requires neighbor sets calculated in the kernel space.
                    kernelNSFUserPresent = true;
                }
            }
            // The embedding NeighborSetFinder objects that will be used to
            // spawn the neighbor set holding objects that are to be passed
            // along to the clusterers at the proper time.
            NeighborSetFinder bigNSF = null;
            NeighborSetFinder bigNSFTest = null;
            NeighborSetFinder bigKernelNSF = null;
            NeighborSetFinder bigKernelNSFTest = null;
            // Counter for garbage collection invocations.
            int memCleanCount = 0;
            // Introducing feature noiseRate, if specified.
            for (float noise = noiseMin; noise <= noiseMax;
                    noise += noiseStep) {
                // Introducing mislabeling, if specified.
                for (float ml = mlMin; ml <= mlMax; ml += mlStep) {
                    if (++memCleanCount % 5 == 0) {
                        System.gc();
                    }
                    // If noiseRate and/or mislabeling are introduced to the
                    // data, we must first make a copy of the original data for
                    // later comparisons, as noiseRate is not introduced to test
                    // data and test labels.
                    if (ml > 0 || noise > 0) {
                        currDSet = originalDSet.copy();
                    } else {
                        currDSet = originalDSet;
                    }
                    if (ml > 0) {
                        // First check if any mislabeling instance weights
                        // were provided, that make certain mislabelings
                        // more probable than others.
                        String weightsPath = null;
                        if (mlWeightsDir != null) {
                            if (!(currCmet instanceof SparseCombinedMetric)) {
                                String metricDir = currCmet.getFloatMetric()
                                        != null
                                        ? currCmet.getFloatMetric().getClass().
                                        getName() : currCmet.getIntegerMetric().
                                        getClass().getName();
                                switch (normType) {
                                    case NONE:
                                        weightsPath = "NO" + File.separator
                                                + metricDir + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case NORM_01:
                                        weightsPath = "NORM01"
                                                + File.separator + metricDir
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case STANDARDIZE:
                                        weightsPath = "STANDARDIZED"
                                                + File.separator + metricDir
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case TFIDF:
                                        weightsPath = "TFIDF"
                                                + File.separator + metricDir
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                }
                            } else {
                                switch (normType) {
                                    case NONE:
                                        weightsPath = "NO" + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case NORM_01:
                                        weightsPath = "NORM01"
                                                + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case STANDARDIZE:
                                        weightsPath = "STANDARDIZED"
                                                + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                    case TFIDF:
                                        weightsPath = "TFIDF"
                                                + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator
                                                + "ml_weights.txt";
                                        break;
                                }
                            }
                            File inWeightFile = new File(mlWeightsDir,
                                    weightsPath);
                            try (BufferedReader br = new BufferedReader(
                                    new InputStreamReader(
                                    new FileInputStream(inWeightFile)));) {
                                String[] weightStrs = br.readLine().split(
                                        " ");
                                float[] mlWeights =
                                        new float[weightStrs.length];
                                for (int i = 0; i < weightStrs.length;
                                        i++) {
                                    mlWeights[i] = Float.parseFloat(
                                            weightStrs[i]);
                                }
                                currDSet.induceWeightProportionalMislabeling(
                                        ml, numCategories, mlWeights);
                            }
                        } else {
                            // Induce the specified mislabeling rate.
                            currDSet.induceMislabeling(ml, numCategories);
                        }
                    }
                    if (noise > 0) {
                        currDSet.addGaussianNoiseToNormalizedCollection(
                                noise, 0.1f);
                    }
                    // We generate a discretized data set, if necessary.
                    if (discreteExists) {
                        currDiscDSet = new DiscretizedDataSet(currDSet);
                        EntropyMDLDiscretizer discretizer =
                                new EntropyMDLDiscretizer(
                                currDSet, currDiscDSet, numCategories);
                        discretizer.discretizeAll();
                        // Or possibly: discretizer.discretizeAllBinary();
                        currDiscDSet.discretizeDataSet(currDSet);
                    }
                    // We keep track of which algorithm is discrete and which
                    // works on continuous data.
                    discrete = new boolean[clustererNames.size()];
                    ArrayList<ClusteringAlg> nonDiscreteAlgs =
                            new ArrayList<>(20);
                    ArrayList<ClusteringAlg> discreteAlgs =
                            new ArrayList<>(20);
                    for (int i = 0; i < clustererNames.size(); i++) {
                        String cName = clustererNames.get(i);
                        ClusteringAlg cInstance;
                        if (algorithmParametrizationMap.containsKey(cName)) {
                            cInstance = getClustererForName(
                                    cName, currDSet, i, kMin, distMat,
                                    null, null, null, numCategories,
                                    algorithmParametrizationMap.get(cName));
                        } else {
                            cInstance = getClustererForName(
                                    cName, currDSet, i, kMin, distMat,
                                    null, null, null, numCategories, null);
                        }
                        if (discrete[i]) {
                            discreteAlgs.add(cInstance);
                        } else {
                            nonDiscreteAlgs.add(cInstance);
                        }
                    }
                    ClusteringAlg[] discreteArray =
                            new ClusteringAlg[discreteAlgs.size()];
                    if (discreteArray.length > 0) {
                        discreteArray = discreteAlgs.toArray(discreteArray);
                    }
                    ClusteringAlg[] nonDiscreteArray =
                            new ClusteringAlg[nonDiscreteAlgs.size()];
                    if (nonDiscreteArray.length > 0) {
                        nonDiscreteArray =
                                nonDiscreteAlgs.toArray(nonDiscreteArray);
                    }
                    // Here we check for which algorithms require distance
                    // matrices and/or kernel matrices to be calculated. If
                    // none require them, they won't be generated, thereby
                    // saving time.
                    for (int i = 0; i < discreteArray.length; i++) {
                        if (discreteArray[i] instanceof
                                DistMatrixUserInterface) {
                            distUserPresentDisc = true;
                            break;
                        }
                    }
                    for (int i = 0; i < nonDiscreteArray.length; i++) {
                        if (nonDiscreteArray[i] instanceof
                                DistMatrixUserInterface) {
                            distUserPresentNonDisc = true;
                            break;
                        }
                    }
                    for (int i = 0; i < discreteArray.length; i++) {
                        if (discreteArray[i] instanceof
                                KernelMatrixUserInterface) {
                            kernelUserPresent = true;
                            break;
                        }
                    }
                    for (int i = 0; i < nonDiscreteArray.length; i++) {
                        if (nonDiscreteArray[i] instanceof
                                KernelMatrixUserInterface) {
                            kernelUserPresent = true;
                            break;
                        }
                    }
                    if (distUserPresentDisc || distUserPresentNonDisc) {
                        // Here a distance matrix path is generated based on the
                        // distance name and the normalization name.
                        String dMatPath = null;
                        if (distancesDir != null) {
                            if (!(currCmet instanceof SparseCombinedMetric)) {
                                switch (normType) {
                                    case NONE:
                                        dMatPath = "NO"
                                                + File.separator
                                                + currCmet.getFloatMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                    case NORM_01:
                                        dMatPath = "NORM01"
                                                + File.separator
                                                + currCmet.getFloatMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                    case STANDARDIZE:
                                        dMatPath =
                                                "STANDARDIZED" + File.separator
                                                + currCmet.getFloatMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                    case TFIDF:
                                        dMatPath = "TFIDF"
                                                + File.separator
                                                + currCmet.getFloatMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                }
                            } else {
                                switch (normType) {
                                    case NONE:
                                        dMatPath = "NO"
                                                + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                    case NORM_01:
                                        dMatPath = "NORM01"
                                                + File.separator
                                                + ((SparseCombinedMetric)
                                                currCmet).getSparseMetric().
                                                getClass().getName()
                                                + File.separator + "dMat.txt";
                                        break;
                                    case STANDARDIZE:
                                        dMatPath =
                                                "STANDARDIZED" + File.separator
                                                + ((SparseCombinedMetric)
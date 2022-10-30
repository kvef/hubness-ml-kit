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
package learning.supervised.evaluation.cv;

import com.google.gson.Gson;
import configuration.BatchClassifierConfig;
import data.neighbors.NSFUserInterface;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.discrete.tranform.EntropyMDLDiscretizer;
import data.representation.sparse.BOWDataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import distances.sparse.SparseCombinedMetric;
import filters.TFIDF;
import ioformat.DistanceMatrixIO;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import ioformat.SupervisedLoader;
import ioformat.results.BatchStatSummarizer;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import learning.supervised.ClassifierFactory;
import learning.supervised.DiscreteClassifier;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.evaluation.ClassifierParametrization;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import networked_experiments.ClassificationResultHandler;
import networked_experiments.HMOpenMLConnector;
import preprocessing.instance_selection.InstanceSelector;
import preprocessing.instance_selection.ReducersFactory;

/**
 * This class implements the functionality for batch testing a series of
 * classification algorithms on a series of datasets, with possible instance
 * selection and / or metric learning, over a series of different neighborhood
 * sizes (for kNN methods), Gaussian feature noise levels and mislabeling levels
 * that can be specified via the configuration file.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BatchClassifierTester {

    // Connector for API calls to OpenML in case when networked experiments are
    // performed and data and training/test splits are obtained from OpenML
    // over the network.
    private HMOpenMLConnector openmlConnector;
    // This directory specification is necessary for registering versioned
    // algorithm source files with OpenML. It is not necessary otherwise, for
    // local experiments.
    private File hubMinerSourceDir;
    // Types of applicable secondary distances.
    public enum SecondaryDistance {

        NONE, SIMCOS, SIMHUB, MP, LS, NICDM
    }
    // Secondary distances are not used by default, unless specified by the
    // user.
    private SecondaryDistance secondaryDistanceType = SecondaryDistance.NONE;
    // Neighborhood size to use for secondary distance calculations. The default
    // is 50, but it can also be directly specified in the experimental
    // configuration file.
    private int secondaryDistanceK = 50;
    // Applicable types of feature normalization.

    public enum Normalization {

        NONE, STANDARDIZE, NORM_01, TFIDF;
    }
    // The normalization type to actually use in the experiments.
    private Normalization normType = Normalization.NONE;
    
    // Paramters for the number of times and folds in cross-validation.
    private int numTimes = 10;
    private int numFolds = 10;
    
    // A range of neighborhood sizes to test for, with default values.
    private int kMin = 5, kMax = 5, kStep = 1;
    // A range of noise and mislabeling rates to test for, with default values.
    private float noiseMin = 0, noiseMax = 0, noiseStep = 1, mlMin = 0,
            mlMax = 0, mlStep = 1;
    // Various files and directories used in the experiment.
    private File inConfigFile, inDir, outDir, currOutDSDir, summaryDir,
            inLabelFile, distancesDir, mlWeightsDir, foldsDir;
    // A list of classifier names to use in the experiment.
    private ArrayList<String> classifierNames = new ArrayList<>(10);
    // Possible parameter value maps to use with certain algorithms.
    HashMap<String, HashMap<String, Object>> algorithmParametrizationMap;
    // Classifier prototype arrays.
    ValidateableInterface[] nonDiscreteArray, discreteArray;
    // List of paths to the datasets that the experiment is to be executed on.
    private ArrayList<String> dsPaths = new ArrayList<>(100);
    // List of CombinedMetric objects for distance calculations that correspond
    // to different datasets. Different datasets might require different
    // metrics to be used, so this is why it is necessary to explicitly specify
    // a metric for each dataset.
    private ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    private ArrayList<Integer>[][][] allDataSetFolds;
    // An alternative specification to the fold specification above, used for
    // OpenML compatibility.
    public ArrayList<Integer>[][][][] trainTestIndexes;
    // Folds to use for testing on the dataset.
    private ArrayList<Integer>[][] dsFolds;
    // Class names are kept track of only for OpenML data sources, as the labels
    // need to be specified in their original form (not just the indexes) when
    // the predictions are uploaded. Class name arrays for non-OpenML data
    // sources will be null in this structure.
    String[][] classNames;
    // DataSet objects.
    private DataSet originalDSet, currDSet;
    // The discretized data object.
    private DiscretizedDataSet currDiscDSet;
    // Current CombinedMetric object.
    private CombinedMetric cmet;
    // The current distance matrix, given as an upper triangular distance matrix
    // so that each row i contains only the distances to j > i, encoded so that
    // distMat[i][j] is the distance between i and i + j + 1. This is the
    // standard matrix format throughout this library.
    private float[][] distMat;
    // A boolean flag array indicating which algorithms require discretized
    // data objects.
    private boolean[] isDiscreteAlgorithm;
    // Whether there is a need for discretizing the data.
    private boolean discreteExists = false;
    // A label array prior to any mislabeling. It is used for evaluation, as
    // classifiers that were trained on mislabeled data need to be evaluated on
    // the exact labels.
    private int[] originalLabels;
    // Integer that is the number of categories in the data.
    private int numCategories;
    // Whether we are in the multi-label experimental mode, where we are testing
    // different representations of the same underlying dataset that has
    // multiple label distributions / classification problems defined on top.
    private boolean multiLabelMode = false;
    // The number of classification problems on the dataset.
    private int numDifferentLabelings = 1;
    // Separator in the label file.
    private String lsep;
    // Whether there are distance users in the continous or discretized case,
    // indicating whether there is a need for calculating the distance matrix
    // or not.
    private boolean distUserPresentDisc = false;
    private boolean distUserPresentNonDisc = false;
    // Whether there is a need to calculate the kNN sets.
    private boolean kNNUserPresent = false;
    // Whether to use the approximate kNN sets and which quality parameter to
    // use.
    private float alphaAppKNNs = 1f;
    private boolean approximateKNNs = false;
    // Current instance selector.
    private InstanceSelector selector = null;
    // Current selection rate.
    private float selectorRate = 0.1f;
    // Hubness estimation mode for the prototypes in instance selection. It can
    // be estimated from all the training data (retained and rejected), which
    // is an unbiased approach to hubness estimation in instance selection - or
    // it can be simply estimated from the selected prototype set, which is
    // simpler but introduces a bias in the estimates.
    private int protoHubnessMode = MultiCrossValidation.PROTO_UNBIASED;
    // The number of threads used for distance matrix and kNN set calculations.
    private int numCommonThreads = 8;
    // OpenML taskID-s and a map that checks whether a particular dataset is a
    // OpenML data source.
    public ArrayList<Integer> openMLTaskIDList;
    public HashMap<Integer, Integer> dataIndexToOpenMLCounterMap =
            new HashMap<>();
    private ExternalExperimentalContext contextObjects; 

    /**
     * Get the human-readable name of a secondary distance type.
     *
     * @param dType SecondaryDistance to get the name of.
     * @return String that is the secondary distance name.
     */
    private String getSecondaryDistanceName(SecondaryDistance dType) {
        switch (dType) {
            case SIMCOS: {
                return "simcos";
            }
            case SIMHUB: {
                return "simhub";
            }
            case MP: {
                return "mutualproximity";
            }
            case LS: {
                return "localscaling";
            }
            case NICDM: {
                return "nicdm";
            }
            default: {
                return "none";
            }
        }
    }

    /**
     * Initialization.
     *
     * @param inConfigFile File that contains the experimental configuration.
     */
    public BatchClassifierTester(File inConfigFile) {
        this.inConfigFile = inConfigFile;
    }

    /**
     * This method runs all the experiments that were specified in the
     * configuration.
     */
    public void runAllTests() throws Exception {
        int datasetIndex = 0;
        DataSet labelCol = null;
        if (multiLabelMode) {
            // Each label array is a column in the label file.
            if (inLabelFile.getPath().endsWith(".arff")) {
                IOARFF aPers = new IOARFF();
                labelCol = aPers.load(inLabelFile.getPath());
            } else if (inLabelFile.getPath().endsWith(".csv")) {
                IOCSV reader = new IOCSV(false, lsep,
                        DataMineConstants.INTEGER);
                labelCol = reader.readData(inLabelFile);
            } else {
                System.out.println("Wrong label format");
                throw new Exception();
            }
            // The number of different classification problems defined on top
            // of the data.
            numDifferentLabelings = labelCol.getNumIntAttr();
        }
        // Iterate over all data representations / datasets.
        for (String dsPath : dsPaths) {
            cmet = dsMetric.get(datasetIndex);
            File dsFile = new File(dsPath);
            originalDSet = SupervisedLoader.loadData(dsFile, multiLabelMode);
            System.out.println("Testing on: " + dsPath);
            // Make all category indexes be in the range [0 .. numCategores - 1]
            originalDSet.standardizeCategories();
            // Perform feature normalization, if specified.
            if (normType != Normalization.NONE) {
                System.out.print("Normalizing features-");
                if (normType == Normalization.NORM_01) {
               
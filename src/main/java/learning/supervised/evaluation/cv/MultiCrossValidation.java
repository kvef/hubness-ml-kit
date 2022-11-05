
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

import combinatorial.Permutation;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.neighbors.approximate.AppKNNGraphLanczosBisection;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.sparse.BOWDataSet;
import data.representation.sparse.BOWInstance;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import distances.secondary.LocalScalingCalculator;
import distances.secondary.MutualProximityCalculator;
import distances.secondary.NICDMCalculator;
import distances.secondary.snd.SharedNeighborCalculator;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.DiscreteCategory;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.evaluation.cv.BatchClassifierTester.SecondaryDistance;
import learning.supervised.interfaces.AutomaticKFinderInterface;
import learning.supervised.interfaces.DiscreteDistToPointsQueryUserInterface;
import learning.supervised.interfaces.DiscreteNeighborPointsQueryUserInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;

/**
 * This class implements the functionality necessary to conduct a
 * cross-validation procedure for evaluating a set of classifiers on a dataset.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MultiCrossValidation {

    // Instance selection specification, in case the instance selection is to be
    // applied during the folds.
    private InstanceSelector dreducer = null;
    private InstanceSelector foldReducer;
    private float selectionRate = 0;
    public final static int PROTO_UNBIASED = 0;
    public final static int PROTO_BIASED = 1;
    private int protoHubnessMode = PROTO_UNBIASED;
    // The number of times and folds to use.
    private int times = 10;
    private int numFolds = 10;
    // The data context.
    private Object dataType;
    // The data instance list.
    private ArrayList data;
    // Number of classes in the data.
    private int numClasses = 2;
    // Execution times.
    public double[] execTimeTotal;
    public double execTimeAllOneRun;
    // Classifier prototypes.
    private ValidateableInterface[] classifiers;
    // Currently active learners (under training or testing).
    private ValidateableInterface[] currClassifierInstances = null;
    // Evaluation structures.
    private boolean keepAllEvaluations = true;
    private ClassificationEstimator[][] estimators = null;
    private ClassificationEstimator[] currEstimator = null;
    private ClassificationEstimator[] averageEstimator = null;
    private float[][] correctPointClassificationArray = null;
    // numAlgorithms x numTimes x dataSize x classSize
    private float[][][][] allLabelAssignments = null;
    // The neighborhood sizes, in case of automatic k calculation within the
    // algorithms.
    private int kMin = 1;
    private int kMax = 20;
    // The neighborhood size, in case a pre-defined value is to be used.
    private int kValue = 5;
    // Whether to use the interval best k search or the fixed k-value mode.
    private int kMode = SINGLE;
    public static final int SINGLE = 0;
    public static final int INTERVAL = 1;
    // Secondary distance type and the secondary neighborhood size.
    private SecondaryDistance secondaryDistanceType = SecondaryDistance.NONE;
    private int secondaryK = 50;
    // The object that does all the distance calculations.
    private CombinedMetric cmet;
    // In case some fold evaluations return errors and are discarded, these
    // numbers are used for averaging the estimators for each algorithm.
    private int[] numFullFolds;
    // These structures are used for the training/test splits and folds in the
    // cross-validation procedure.
    private ArrayList<Integer>[][] allFolds = null;
    private boolean foldsLoaded = false;
    private ArrayList[] dataFolds = null;
    private ArrayList currentTraining;
    private ArrayList<Integer>[] foldIndexes = null;
    private ArrayList<Integer> currentTrainingIndexes;
    private ArrayList<Integer> currentTestIndexes;
    // This is used for external train/test index setting for OpenML
    // compatibility. In that case, the generated or disk-loaded folds are not
    // used, but ignored. The 0th index of trainTestIndexes[i][j] contains
    // i-times j-fold train indexes and the 1st index contains the test indexes.
    private ArrayList<Integer>[][][] trainTestIndexes;
    // In case of instance selection.
    private ArrayList<Integer> currentPrototypeIndexes;
    // A separate label array is kept to be used in case of mislabeling data
    // experiments.
    private boolean validateOnExternalLabels = false;
    private int[] testLabelArray = null;
    // The total distance matrix, as an upper triangular matrix.
    public float[][] totalDistMat = null;
    // Flags indicating whether there are users of the distance matrix or the
    // total kNN sets on the training data.
    private boolean distUserPresent = false;
    private boolean nsfUserPresent = false;
    // The kNN structures.
    private NeighborSetFinder bigNSF;
    private NeighborSetFinder nsfCurrent = null;
    private int[][] testPointNeighbors = null;
    private boolean approximateNNs = false;
    private float alphaAppKNN = 1f;
    // The number of threads used for distance matrix and kNN set calculations.
    private int numCommonThreads = 8;
    // If kNN sets and/or distance matrix are available externally.
    ExternalExperimentalContext contextObjects;

    /**
     * The default constructor.
     */
    public MultiCrossValidation() {
    }

    /**
     * Initialization.
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public MultiCrossValidation(int times, int folds, int numClasses) {
        this.times = times;
        this.numFolds = folds;
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     * @param totalDistMat float[][] that is the upper triangular distance
     * matrix on the training data.
     */
    public MultiCrossValidation(int times, int folds, int numClasses,
            float[][] totalDistMat) {
        this.times = times;
        this.numFolds = folds;
        this.numClasses = numClasses;
        this.totalDistMat = totalDistMat;
    }

    /**
     * Initialization with a single classifier.
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     * @param dataType Object that is the data context.
     * @param data ArrayList of data instances.
     * @param classifier ValidateableInterface signifying the classifier to
     * evaluate.
     */
    public MultiCrossValidation(int times, int folds, int numClasses,
            Object dataType, ArrayList data, ValidateableInterface classifier) {
        this.times = times;
        this.numFolds = folds;
        this.dataType = dataType;
        this.data = data;
        this.classifiers = new ValidateableInterface[1];
        classifiers[0] = classifier;
        currClassifierInstances = new ValidateableInterface[1];
        execTimeTotal = new double[1];
        this.numClasses = numClasses;
    }

    /**
     * Initialization with a single classifier..
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     * @param dataType Object that is the data context.
     * @param data ArrayList of data instances.
     * @param classifier ValidateableInterface signifying the classifier to
     * evaluate.
     * @param totalDistMat
     */
    public MultiCrossValidation(int times, int folds, int numClasses,
            Object dataType, ArrayList data, ValidateableInterface classifier,
            float[][] totalDistMat) {
        this.times = times;
        this.numFolds = folds;
        this.dataType = dataType;
        this.data = data;
        this.classifiers = new ValidateableInterface[1];
        classifiers[0] = classifier;
        currClassifierInstances = new ValidateableInterface[1];
        execTimeTotal = new double[1];
        this.numClasses = numClasses;
        this.totalDistMat = totalDistMat;
    }

    /**
     * Initialization.
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     * @param dataType Object that is the data context.
     * @param data ArrayList of data instances.
     * @param classifiers ValidateableInterface[] classifiers signifying an
     * array of classifiers to evaluate.
     */
    public MultiCrossValidation(int times, int folds, int numClasses,
            Object dataType, ArrayList data,
            ValidateableInterface[] classifiers) {
        this.times = times;
        this.numFolds = folds;
        this.dataType = dataType;
        this.data = data;
        this.classifiers = classifiers;
        currClassifierInstances = new ValidateableInterface[classifiers.length];
        execTimeTotal = new double[classifiers.length];
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param times Integer that is the number of times to repeat the fold split
     * and evaluation process.
     * @param folds Integer that is the number of folds to generate in each
     * repetition.
     * @param numClasses Integer that is the number of classes in the data.
     * @param dataType Object that is the data context.
     * @param data ArrayList of data instances.
     * @param classifiers ValidateableInterface[] classifiers signifying an
     * array of classifiers to evaluate.
     * @param totalDistMat float[][] that is the upper triangular distance
     * matrix on the training data.
     */
    public MultiCrossValidation(int times, int folds, int numClasses,
            Object dataType, ArrayList data,
            ValidateableInterface[] classifiers, float[][] totalDistMat) {
        this.times = times;
        this.numFolds = folds;
        this.dataType = dataType;
        this.data = data;
        this.classifiers = classifiers;
        currClassifierInstances = new ValidateableInterface[classifiers.length];
        execTimeTotal = new double[classifiers.length];
        this.numClasses = numClasses;
        this.totalDistMat = totalDistMat;
    }
    
    /**
     * @param contextObjects ExternalExperimentalContext object potentially
     * holding pre-calculated re-usable kNN sets and/or the distance matrix.
     */
    public void setExternalContext(ExternalExperimentalContext contextObjects) {
        this.contextObjects = contextObjects;
    }
    
    /**
     * This method sets the train/test indexes explicitly and is used for OpenML
     * compatibility. It should not be used otherwise, but standard split/fold
     * configurations should be provided, as that is a more fine-grained control
     * of what goes on in cross-validation. This explicit train/test
     * specification might be more flexible, but also allows some less favorable
     * configurations if no checks are performed.
     * 
     * @param trainTestIndexes ArrayList<Integer>[][][] representing the train
     * and test index lists for all repetitions and folds.
     */
    public void setTrainTestIndexes(ArrayList<Integer>[][][] trainTestIndexes) {
        this.trainTestIndexes = trainTestIndexes;
    }
    
    /**
     * This method gives the used train and test indexes, if the folds have been
     * tracked or the train/test configuration was set externally.
     * 
     * @return ArrayList<Integer>[][][] that are the train / test indexes.
     */
    public ArrayList<Integer>[][][] getTrainTestIndexes() {
        if (trainTestIndexes == null && allFolds != null) {
            trainTestIndexes = new ArrayList[times][numFolds][2];
            for (int i = 0; i < times; i++) {
                for (int j = 0; j < numFolds; j++) {
                    trainTestIndexes[i][j][0] = new ArrayList<>();
                    trainTestIndexes[i][j][1] = new ArrayList<>();
                }
            }
            for (int i = 0; i < times; i++) {
                for (int j = 0; j < numFolds; j++) {
                    // Test indexes.
                    trainTestIndexes[i][j][1] = allFolds[i][j];
                    for (int k = 0; k < j; k++) {
                        trainTestIndexes[i][j][0].addAll(allFolds[i][j]);
                    }
                    for (int k = j + 1; k < numFolds; k++) {
                        trainTestIndexes[i][j][0].addAll(allFolds[i][j]);
                    }
                }
            }
        }
        return trainTestIndexes;
    }
    
    /**
     * Sets the pre-computed folds to the cross-validation procedure. An
     * exception is thrown if the dimensions of the array of lists of indexes do
     * not correspond to the number of times and folds that have been set in
     * this cross-validation context.
     * 
     * @param allFolds ArrayList<Integer>[][] representing all folds in all
     * repetitions of the cross-validation procedure, by giving indexes of the
     * points within the dataset in each split.
     * @throws Exception if the fold array dimensionality does not correspond
     * to the times x numFolds that are set within this cross-validation
     * context.
     */
    public void setAllFolds(ArrayList<Integer>[][] allFolds) throws Exception {
        this.allFolds = allFolds;
        if (allFolds != null) {
            if (allFolds.length != times) {
                throw new Exception("Bad fold format, not matching the "
                        + "specified number of repetitions.");
            } else {
                for (int repIndex = 0; repIndex < times; repIndex++) {
                    if (allFolds[repIndex] == null ||
                            allFolds[repIndex].length != numFolds) {
                        throw new Exception("Bad fold format. The number of"
                                + "folds in some repetitions does not match"
                                + "the specified number of folds in this"
                                + "cross-validation context.");
                    }
                }
            }
            foldsLoaded = true;
        }
    }
    
    /**
     * @return ArrayList<Integer>[][] representing all folds in all repetitions 
     * of the cross-validation procedure, by giving indexes of the points within
     * the dataset in each split.
     */
    public ArrayList<Integer>[][] getAllFolds() {
        return allFolds;
    }
    
    /**
     * This sets the number of threads to use for distance matrix and kNN set
     * calculations.
     * 
     * @param numCommonThreads Integer that is the number of threads to use for
     * distance matrix and kNN set calculations.
     */
    public void useMultipleCommonThreads(int numCommonThreads) {
        this.numCommonThreads = numCommonThreads;
    }

    /**
     * @param protoHubnessMode Integer code that indicates which prototype
     * hubness estimation mode to use, whether to use the biased simple approach
     * or to estimate the prototype occurrence frequencies on the rejected
     * points as well.
     */
    public void setProtoHubnessMode(int protoHubnessMode) {
        this.protoHubnessMode = protoHubnessMode;
    }

    /**
     * Sets the instance selection structures.
     * @param dreducer InstanceSelector that is to be used for data reduction.
     * @param selectionRate Float that is the instance selection rate.
     */
    public void setDataReducer(InstanceSelector dreducer, float selectionRate) {
        this.dreducer = dreducer;
        this.selectionRate = selectionRate;
    }

    /**
     * @return InstanceSelector that is being used for data reduction.
     */
    public InstanceSelector getDataReducer() {
        return dreducer;
    }

    /**
     * @return float[][] representing the point classification precision for
     * each tested algorithm.
     */
    public float[][] getPerPointClassificationPrecision() {
        return correctPointClassificationArray;
    }
    
    /**
     * @return int[][][][] representing all fuzzy label assignments given per
     * algorithm for all repetitions of the CV framework.
     */
    public float[][][][] getAllFuzzyLabelAssignments() {
        return allLabelAssignments;
    }

    /**
     * @param secondaryDistanceType SecondaryDistance that is to be used.
     * @param secondaryK Integer that is the neighborhood size to use when
     * calculating the secondary distances.
     */
    public void useSecondaryDistances(SecondaryDistance secondaryDistanceType,
            int secondaryK) {
        this.secondaryDistanceType = secondaryDistanceType;
        this.secondaryK = secondaryK;
    }

    /**
     * @param cmet CombinedMetric object for distance calculations.
     */
    public void setCombinedMetric(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * @param k Integer that is the neighborhood size to test for. 
     */
    public void setKValue(int k) {
        this.kValue = k;
        this.kMin = k;
        this.kMax = k;
        kMode = SINGLE;
    }

    /**
     * In case the algorithms are expected to automatically determine the
     * optimal k-value.
     * @param kMin Integer that is the lower range value for the neighborhood
     * size.
     * @param kMax Integer that is the upper range value for the neighborhood
     * size.
     */
    public void setAutoKRange(int kMin, int kMax) {
        this.kMin = kMin;
        this.kMax = kMax;
        kMode = INTERVAL;
    }

    /**
     * @param approximate Boolean flag indicating whether to use the approximate
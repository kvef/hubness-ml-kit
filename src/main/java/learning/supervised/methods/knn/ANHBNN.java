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
package learning.supervised.methods.knn;

import algref.Author;
import algref.ConferencePublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.awt.Point;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.evaluation.cv.MultiCrossValidation;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;
import util.BasicMathUtil;

/**
 * This class implements the Augmented Naive Hubness-Bayesian k-Nearest Neighbor
 * method that was proposed in the following paper: Hub Co-occurrence Modeling
 * for Robust High-dimensional kNN Classification, by Nenad Tomasev and Dunja
 * Mladenic, which was presented at ECML/PKDD 2013 in Prague. The algorithm is
 * an extension of NHBNN, the Naive Bayesian re-interpretation of the k-nearest
 * neighbor rule. It incorporates the Hidden Naive Bayes model for kNN
 * classification by modeling the class-conditional neighbor co-occurrence
 * probabilities.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ANHBNN extends Classifier implements DistMatrixUserInterface,
        NSFUserInterface, DistToPointsQueryUserInterface,
        NeighborPointsQueryUserInterface, Serializable {
    
    private static final long serialVersionUID = 1L;

    // The anti-hub cut-off parameter.
    private int thetaValue = 0;
    // The current neighborhood size.
    private int k = 5;
    // Object that holds the kNN sets.
    private NeighborSetFinder nsf = null;
    // Training dataset to infer the model from.
    private DataSet trainingData = null;
    // The number of classes in the data.
    private int numClasses = 0;
    private ArrayList<Point> coOccurringPairs;
    // One map for each class, that maps a long value obtained from two index
    // values to a value that is the current co-occurrence count.
    private HashMap<Long, Integer>[] coDependencyMaps;
    // Mutual information between neighbor pairs.
    private HashMap<Long, Double> mutualInformationMap;
    // Class-conditional neighbor occurrence counts.
    private float[][] classDataKNeighborRelation = null;
    // Class priors.
    private float[] classPriors = null;
    // Class frequencies.
    private float[] classFreqs = null;
    // Float value that is the Laplace estimator for probability distribution
    // smoothing. There are two values, a smaller and a larger one, for
    // smoothing different types of distributions.
    private float laplaceEstimatorSmall = 0.000000000001f;
    private float laplaceEstimatorBig = 0.1f;
    // Neighbor occurrence frequencies.
    private int[] neighbOccFreqs = null;
    // Non-homogeneity of reverse neighbor sets.
    private float[] rnnImpurity = null;
    private double[][] classConditionalSelfInformation = null;
    private float[][] classToClassPriors = null;
    private float[][][] classCoOccurrencesInNeighborhoodsOfClasses;
    private float[][] distMat = null;
    private boolean noRecalc = false;
    private int dataSize;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("thetaValue", "Anti-hub cut-off point for treating"
                + "anti-hubs as a special case.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setConferenceName("European Conference on Machine Learning");
        pub.addAuthor(Author.NENAD_TOMASEV);
        pub.addAuthor(Author.DUNJA_MLADENIC);
        pub.setTitle("Hub Co-occurrence Modeling for Robust High-dimensional "
                + "kNN Classification");
        pub.setYear(2013);
        pub.setStartPage(643);
        pub.setEndPage(659);
        pub.setPublisher(Publisher.SPRINGER);
        return pub;
    }

    @Override
    public void setDistMatrix(float[][] distMatrix) {
        this.distMat = distMatrix;
    }

    @Override
    public float[][] getDistMatrix() {
        return distMat;
    }

    @Override
    public String getName() {
        return "ANHBNN";
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public void noRecalcs() {
        noRecalc = true;
    }

    /**
     * The default constructor.
     */
    public ANHBNN() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public ANHBNN(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public ANHBNN(int k, CombinedMetric cmet) {
        this.k = k;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public ANHBNN(int k, CombinedMetric cmet, int numClasses) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;

    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float value used as a Laplace estimator for
     * probability estimate smoothing in probability distributions.
     */
    public ANHBNN(int k, float laplaceEstimator) {
        this.k = k;
        this.laplaceEstimatorSmall = laplaceEstimator;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float value used as a Laplace estimator for
     * probability estimate smoothing in probability distributions.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public ANHBNN(int k, float laplaceEstimator, CombinedMetric cmet) {
        this.k = k;
        this.laplaceEstimatorSmall = laplaceEstimator;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float value used as a Laplace estimator for
     * probability estimate smoothing in probability distributions.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public ANHBNN(int k, float laplaceEstimator, CombinedMetric cmet,
            int numClasses) {
        this.k = k;
        this.laplaceEstimatorSmall = laplaceEstimator;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object used for model training.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public ANHBNN(DataSet dset, int numClasses, CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object used for model training.
     * @param numClasses Integer that is the number of classes in the data.
     * @param nsf NeighborSetFinder object for kNN calculations.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public ANHBNN(DataSet dset, int numClasses, NeighborSetFinder nsf,
            CombinedMetric cmet, int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        this.nsf = nsf;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param categories Category[] representing the training data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public ANHBNN(Category[] categories, CombinedMetric cmet, int k) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1
                    && categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // Instances are not embedded in the internal data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int i = 0; i < categories[cIndex].size(); i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(i));
            }
        }
        setCombinedMetric(cmet);
        this.k = k;
        numClasses = trainingData.countCategories();
    }

    /**
     * @param laplaceEstimator Float value used as a Laplace estimator for
     * probability estimate smoothing in probability distributions.
     */
    public void setLaplaceEstimator(float laplaceEstimator) {
        this.laplaceEstimatorSmall = laplaceEstimator;
    }

    @Override
    public void setClasses(Category[] categories) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            totalSize += categories[cIndex].size();
            if (indexFirstNonEmptyClass == -1
                    && categories[cIndex].size() > 0) {
                indexFirstNonEmptyClass = cIndex;
            }
        }
        // Instances are not embedded in the internal data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int i = 0; i < categories[cIndex].size(); i++) {
                categories[cIndex].getInstance(i).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(i));
            }
        }
        numClasses = trainingData.countCategories();
    }

    /**
     * @param trainingData DataSet object to train the model on.
     */
    public void setTrainingSet(DataSet trainingData) {
        this.trainingData = trainingData;
    }

    /**
     * @param numClasses Integer that is the number of classes in the data.
     */
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    /**
     * @return Integer that is the number of classes in the data.
     */
    public int getNumClasses() {
        return numClasses;
    }

    /**
     * @return Integer that is the neighborhood size used in calculations.
     */
    public int getK() {
        return k;
    }

    /**
     * @param k Integer that is the neighborhood size used in calculations.
     */
    public void setK(int k) {
        this.k = k;
    }

    @Override
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    @Override
    public NeighborSetFinder getNSF() {
        return nsf;
    }

    /**
     * Calculate the neighbor sets, if not already calculated.
     *
     * @throws Exception
     */
    public void calculateNeighborSets() throws Exception {
        if (distMat == null) {
            nsf = new NeighborSetFinder(trainingData, getCombinedMetric());
            nsf.calculateDistances();
        } else {
            nsf = new NeighborSetFinder(trainingData, distMat,
                    getCombinedMetric());
        }
        nsf.calculateNeighborSets(k);
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        ANHBNN classifierCopy = new ANHBNN(k, laplaceEstimatorSmall,
                getCombinedMetric(), numClasses);
        classifierCopy.noRecalc = noRecalc;
        classifierCopy.thetaValue = thetaValue;
        return classifierCopy;
    }

    /**
     * Calculates the mutual information between two neighbors according to
     * their occurrences and co-occurrences.
     *
     * @param min
     * @param max
     * @return
     */
    private double calculateMutualInformation(int lowerIndex, long upperIndex) {
        // Transform to the encoding used in the hash maps.
        long concat = (upperIndex << 32) | (lowerIndex & 0XFFFFFFFFL);
        int size = trainingData.size();
        if (mutualInformationMap.containsKey(concat)) {
            // If it has already been queried and calculated before, just load
            // the existing result.
            return mutualInformationMap.get(concat);
        } else {
            // Calculate the mutual information from mutual and individual
            // occurrence counts.
            double bothOccurFactor = 0;
            double firstOccursFactor = 0;
            double secondOccursFactor = 0;
            double noneOccursFactor = 0;
            int cooccFreq;
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (coDependencyMaps[cIndex].containsKey(concat)) {
                    cooccFreq = coDependencyMaps[cIndex].get(concat);
                } else {
                    cooccFreq = 0;
                }
                // The formulas are a bit complicated. For more detail, look up
                // the original paper, as it is freely available online.
                bothOccurFactor += ((double) (cooccFreq) / (double) size)
                        * BasicMathUtil.log2(((double) (cooccFreq
                        + laplaceEstimatorSmall) / (double) classFreqs[cIndex]
                        + laplaceEstimatorSmall) / (((classDataKNeighborRelation[
                        cIndex][lowerIndex] + laplaceEstimatorSmall)
                        / ((double) classFreqs[cIndex] + laplaceEstimatorSmall))
                        * ((classDataKNeighborRelation[cIndex][(int) upperIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall))));
                firstOccursFactor += ((double) (classDataKNeighborRelation[
                        cIndex][lowerIndex] - cooccFreq) / (double) size)
                        * BasicMathUtil.log2(((double) (
                        classDataKNeighborRelation[cIndex][lowerIndex]
                        - cooccFreq + laplaceEstimatorSmall) /
                        ((double) classFreqs[cIndex] + laplaceEstimatorSmall))
                        / (((classDataKNeighborRelation[cIndex][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall)) * (1
                        - ((classDataKNeighborRelation[cIndex][(int) upperIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall)))));
                secondOccursFactor += ((double) (classDataKNeighborRelation[
                        cIndex][(int) upperIndex] - cooccFreq) / (double) size)
                        * BasicMathUtil.log2(((double) (
                        classDataKNeighborRelation[cIndex][(int) upperIndex]
                        - cooccFreq + laplaceEstimatorSmall) /
                        ((double) classFreqs[cIndex] + laplaceEstimatorSmall)) /
                        (((classDataKNeighborRelation[cIndex][(int) upperIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall)) * (1 -
                        ((classDataKNeighborRelation[cIndex][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall)))));
                noneOccursFactor += ((double) (classFreqs[cIndex]
                        - classDataKNeighborRelation[cIndex][lowerIndex]
                        - classDataKNeighborRelation[cIndex][(int) upperIndex]
                        + cooccFreq) / (double) size) * BasicMathUtil.log2(
                        ((double) (classFreqs[cIndex]
                        - classDataKNeighborRelation[cIndex][lowerIndex]
                        - classDataKNeighborRelation[cIndex][(int) upperIndex]
                        + cooccFreq + laplaceEstimatorSmall)
                        / ((double) classFreqs[cIndex] + laplaceEstimatorSmall))
                        / ((1 - ((classDataKNeighborRelation[cIndex][
                        (int) upperIndex] + laplaceEstimatorSmall)
                        / ((double) classFreqs[cIndex] +
                        laplaceEstimatorSmall)))
                        * (1 - ((classDataKNeighborRelation[cIndex][lowerIndex]
                        + laplaceEstimatorSmall) / ((double) classFreqs[cIndex]
                        + laplaceEstimatorSmall)))));
            }
            double mutualInformation = bothOccurFactor + firstOccursFactor
                    + secondOccursFactor + noneOccursFactor;
            mutualInformationMap.put(concat, mutualInformation);
            return mutualInformation;
        }
    }

    /**
     * Calculates the entropies of the reverse neighbor sets.
     *
     * @param kneighbors int[][] of k-nearest neighbors for all training
     * instances.
     * @return float[] of reverse neighbor set entropies for all training
     * instances.
     */
    private float[] calculateReverseNeighborEntropies(int[][] kneighbors) {
        // Category frequencies in the reverse neighbor set of a particular
        // point.
        float[] categoryFrequencies = new float[numClasses];
        float[] reverseNeighborEntropies = new float[trainingData.size()];
        ArrayList<Integer>[] reverseNeighbors =
                new ArrayList[trainingData.size()];
        for (int i = 0; i < trainingData.size(); i++) {
            reverseNeighbors[i] = new ArrayList<>(10 * k);
        }
        for (int i = 0; i < trainingData.size(); i++) {
            for (int kIndex = 0; kIndex < k; kIndex++) {
                reverseNeighbors[kneighbors[i][kIndex]].add(i);
            }
        }
        float ratio;
        for (int i = 0; i < reverseNeighborEntropies.length; i++) {
            if (reverseNeighbors[i].size() <= 1) {
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    categoryFrequencies[cIndex] = 0;
                }
                reverseNeighborEntropies[i] = 0;
                continue;
            }
            for (int j = 0; j < reverseNeighbors[i].size(); j++) {
                int currClass = trainingData.getInstance(
                        reverseNeighbors[i].get(j)).getCategory();
                if (currClass >= 0) {
                    categoryFrequencies[currClass]++;
                }
            }
            // Calculate the entropy.
            for (int j = 0; j < categoryFrequencies.length; j++) {
                if (categoryFrequencies[j] > 0) {
                    ratio = categoryFrequencies[j]
                            / (float) reverseNeighbors[i].size();
                    reverseNeighborEntropies[i] -=
                            ratio * BasicMathUtil.log2(ratio);
                }
            }
            // Nullify the category frequencies array for next use.
            for (int j = 0; j < numClasses; j++) {
                categoryFrequencies[j] = 0;
            }
        }
        return reverseNeighborEntropies;
    }

    @Override
    public void train() throws Exception {
        if (k <= 0) {
            // If the neighborhood size was not specified, use the default
            // value. TODO: implement automatic parameter selection.
            k = 10;
        }
        if (nsf == null) {
            calculateNeighborSets();
        }
        dataSize = trainingData.size();
        // Calculate class priors.
        classPriors = trainingData.getClassPriors();
        classFreqs = trainingData.getClassFrequenciesAsFloatArray();
        classToClassPriors = new float[numClasses][numClasses];
        // Get the neighbor occurrence frequencies.
        neighbOccFreqs = nsf.getNeighborFrequencies();
        // Calculate the entropies of the reverse neighbor sets.
        nsf.calculateReverseNeighborEntropies(numClasses);
        rnnImpurity = nsf.getReverseNeighborEntropies();
        // The list of co-occurring pairs of points.
        coOccurringPairs = new ArrayList<>(dataSize);
        // The kNN sets.
        int[][] kneighbors = nsf.getKNeighbors();
        // The map for storing the co-occurrence counts. 
        coDependencyMaps = new HashMap[numClasses];
        mutualInformationMap = new HashMap<>(dataSize);
        classConditionalSelfInformation = new double[dataSize][numClasses];
        classCoOccurrencesInNeighborhoodsOfClasses =
                new float[numClasses][numClasses][numClasses];
        // Initialize the maps.
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            coDependencyMaps[cIndex] = new HashMap<>(dataSize);

        }
        long concat; // Used to encode neighbor pairs to a single hashable
        // value.
        int lowerIndex;
        long upperIndex;
        int queryClass;
        int currFreq;
        classDataKNeighborRelation = new float[numClasses][trainingData.size()];
        for (int i = 0; i < neighbOccFreqs.length; i++) {
            // Get the class context of the query.
            queryClass = trainingData.getLabelOf(i);
            // Each point is considered
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
     * @param trainingData
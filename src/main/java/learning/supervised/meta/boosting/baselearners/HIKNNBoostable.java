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
package learning.supervised.meta.boosting.baselearners;

import algref.Author;
import algref.ConferencePublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import learning.supervised.meta.boosting.BoostableClassifier;
import util.ArrayUtil;
import util.BasicMathUtil;

/**
 * This class implements the HIKNN algorithm that was proposed in the paper
 * titled: "Nearest Neighbor Voting in High Dimensional Data: Learning from Past
 * Occurrences" published in Computer Science and Information Systems in 2011.
 * The algorithm is an extension of h-FNN that gives preference to rare neighbor
 * points and uses some label information. This is an extension that supports
 * instance weights for boosting.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HIKNNBoostable extends BoostableClassifier implements
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    private int k = 5;
    // NeighborSetFinder object for kNN calculations.
    private NeighborSetFinder nsf = null;
    private DataSet trainingData = null;
    private int numClasses = 0;
    // The distance weighting parameter.
    private float mValue = 2;
    private double[][] classDataKNeighborRelation = null;
    // Information contained in the neighbors' labels.
    private float[] labelInformationFactor = null;
    // The prior class distribution.
    private float[] classPriors = null;
    private float laplaceEstimator = 0.001f;
    private int[] neighborOccurrenceFreqs = null;
    private double[] neighborOccurrenceFreqsWeighted = null;
    // The distance matrix.
    private float[][] distMat;
    private boolean noRecalc = true;
    // Boosting weights.
    private double[] instanceWeights;
    private double[][] instanceLabelWeights;
    // Boosting mode.
    public static final int B1 = 0;
    public static final int B2 = 1;
    private int boostingMode = B1;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("mValue", "Exponent for distance weighting. Defaults"
                + " to 2.");
        paramMap.put("boostingMode", "Type of re-weighting procedure.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setTitle("Boosting for Vote Learning in High-Dimensional k-Nearest "
                + "Neighbor Classification");
        pub.addAuthor(Author.NENAD_TOMASEV);
        pub.setConferenceName("Workshop on High-Dimensional Data Mining at the "
                + "International Conference on Data Mining");
        pub.setYear(2014);
        pub.setPublisher(Publisher.IEEE);
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    /**
     * @param boostingMode Integer that is the current boosting mode: B1 or B2.
     */
    public void setBoostingMode(int boostingMode) {
        this.boostingMode = boostingMode;
    }

    @Override
    public void setTotalInstanceWeights(double[] instanceWeights) {
        this.instanceWeights = instanceWeights;
    }

    @Override
    public void setMisclassificationCostDistribution(
            double[][] instanceLabelWeights) {
        this.instanceLabelWeights = instanceLabelWeights;
    }

    @Override
    public String getName() {
        return "HIKNN";
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
    public void noRecalcs() {
        noRecalc = true;
    }

    /**
     * The default constructor.
     */
    public HIKNNBoostable() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public HIKNNBoostable(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HIKNNBoostable(int k, CombinedMetric cmet) {
        this.k = k;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes.
     */
    public HIKNNBoostable(int k, CombinedMetric cmet, int numClasses) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes.
     * @param boostingMode Integer that is the current boosting mode.
     */
    public HIKNNBoostable(int k, CombinedMetric cmet, int numClasses,
            int boostingMode) {
        this.k = k;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
        this.boostingMode = boostingMode;

    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
     */
    public HIKNNBoostable(int k, float laplaceEstimator) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public HIKNNBoostable(int k, float laplaceEstimator, CombinedMetric cmet) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClasses Integer that is the number of classes.
     */
    public HIKNNBoostable(int k, float laplaceEstimator, CombinedMetric cmet,
            int numClasses) {
        this.k = k;
        this.laplaceEstimator = laplaceEstimator;
        setCombinedMetric(cmet);
        this.numClasses = numClasses;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the training data.
     * @param numClasses Integer that is the number of classes.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HIKNNBoostable(DataSet dset, int numClasses, CombinedMetric cmet,
            int k) {
        trainingData = dset;
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object that is the training data.
     * @param numClasses Integer that is the number of classes.
     * @param nsf NeighborSetFinder object for kNN calculations.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HIKNNBoostable(DataSet dset, int numClasses, NeighborSetFinder nsf,
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
     * @param categories Category[] of classes to train on.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Integer that is the neighborhood size.
     */
    public HIKNNBoostable(Category[] categories, CombinedMetric cmet, int k) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int i = 0; i < categories.length; i++) {
            totalSize += categories[i].size();
            if (indexFirstNonEmptyClass == -1 && categories[i].size() > 0) {
                indexFirstNonEmptyClass = i;
            }
        }
        // As this is an internal DataSet object, data instances will not have
        // it set as their data context.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames = categories[indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int i = 0; i < categories.length; i++) {
            for (int j = 0; j < categories[i].size(); j++) {
                categories[i].getInstance(j).setCategory(i);
                trainingData.addDataInstance(categories[i].getInstance(j));
            }
        }
        setCombinedMetric(cmet);
        this.k = k;
    }

    /**
     * @param laplaceEstimator Float that is the Laplace estimator for smoothing
     * the probability distributions.
     */
    public void setLaplaceEstimator(float laplaceEstimator) {
        this.laplaceEstimator = laplaceEstimator;
    }

    @Override
    public void setClasses(Category[] categories) {
        int totalSize = 0;
        int indexFirstNonEmptyClass = -1;
        for (int i = 0; i < categories.length; i++) {
            totalSize += categories[i].size();
            if (indexFirstNonEmptyClass == -1 && categories[i].size() > 0) {
                indexFirstNonEmptyClass = i;
            }
        }
        // An internal training data representation.
        trainingData = new DataSet();
        trainingData.fAttrNames = categories[
                indexFirstNonEmptyClass].
                getInstance(0).getEmbeddingDataset().fAttrNames;
        trainingData.iAttrNames =
                categories[indexFirstNonEmptyClass]
                .getInstance(0).getEmbeddingDataset().iAttrNames;
        trainingData.sAttrNames =
                categories[indexFirstNonEmptyClass].getInstance(0).
                getEmbeddingDataset().sAttrNames;
        trainingData.data = new ArrayList<>(totalSize);
        for (int cIndex = 0; cIndex < categories.length; cIndex++) {
            for (int j = 0; j < categories[cIndex].size(); j++) {
                categories[cIndex].getInstance(j).setCategory(cIndex);
                trainingData.addDataInstance(categories[cIndex].getInstance(j));
            }
        }
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
        HIKNNBoostable classifierCopy = new HIKNNBoostable(k, laplaceEstimator,
                getCombinedMetric(), numClasses);
        classifierCopy.noRecalc = noRecalc;
        classifierCopy.boostingMode = boostingMode;
        return classifierCopy;
    }

    @Override
    public void train() throws Exception {
        if (nsf == null) {
            // If the kNN sets have not been provided, calculate them.
            calculateNeighborSets();
        }
        // Find the class priors.
        classPriors = trainingData.getClassPriors();
        if (!noRecalc) {
            nsf.recalculateStatsForSmallerK(k);
        }
        // Set default values for instance weights if none have been provided.
        if (instanceWeights == null) {
            instanceWeights = new double[trainingData.size()];
            Arrays.fill(instanceWeights, 1d);
        }
        if (instanceLabelWeights == null) {
            instanceLabelWeights = new double[trainingData.size()][numClasses];
        }
        // Get the neighbor occurrence frequenc
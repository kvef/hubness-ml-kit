
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
package learning.supervised.methods.discrete;

import algref.Author;
import algref.ConferencePublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.NeighborSetFinder;
import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import learning.supervised.DiscreteCategory;
import learning.supervised.DiscreteClassifier;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.AutomaticKFinderInterface;

/**
 * This class implements the k-nearest neighbor Naive Bayes algorithm, KNNNB,
 * that learns a Naive Bayes model on the restriction of the data on the kNN
 * set. Only the features from the k-neighbors are used when learning a
 * probabilistic model.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KNNNB extends DiscreteClassifier
        implements AutomaticKFinderInterface, Serializable {

    private static final long serialVersionUID = 1L;
    public static final int DEFAULT_NEIGHBORHOOD_SIZE = 20;
    private float laplaceEstimator = 1f;
    // Neighborhood size.
    private int k = DEFAULT_NEIGHBORHOOD_SIZE;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("laplaceEstimator", "Laplace estimator for smoothing."
                + "Defaults to 1.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setConferenceName("Fuzzy Systems and Knowledge Discovery");
        pub.addAuthor(new Author("Liangxiao", "Jiang"));
        pub.addAuthor(new Author("Zhihua", "Cai"));
        pub.addAuthor(new Author("Dianhong", "Wang"));
        pub.addAuthor(new Author("Siwei", "Jiang"));
        pub.setTitle("Survey of Improving K-Nearest-Neighbor for "
                + "Classification");
        pub.setYear(2007);
        pub.setStartPage(679);
        pub.setEndPage(683);
        pub.setPublisher(Publisher.IEEE);
        return pub;
    }

    @Override
    public String getName() {
        return "KNNNB";
    }

    @Override
    public void findK(int kMin, int kMax) throws Exception {
        DiscretizedDataSet discDSet = getDataType();
        int numClasses = getClasses().length;
        float currMaxAcc = -1f;
        int currMaxK = 0;
        KNNNB classifier;
        ArrayList<DiscretizedDataInstance> data = discDSet.data;
        // Generate some folds to test on, for finding the best parameter
        // configuration.
        Random randa = new Random();
        ArrayList[] dataFolds = null;
        ArrayList[] dcDataFolds = null;
        ArrayList currentTraining;
        ArrayList currentDCTraining;
        ArrayList[] foldIndexes = null;
        ArrayList<Integer> currentIndexes;
        ArrayList<Integer> currentTest;
        // In the default implementation, make one training and test set.
        int folds = 2;
        float choice;
        boolean noEmptyFolds = false;
        while (!noEmptyFolds) {
            dataFolds = new ArrayList[folds];
            dcDataFolds = new ArrayList[folds];
            foldIndexes = new ArrayList[folds];
            for (int foldIndex = 0; foldIndex < folds; foldIndex++) {
                dataFolds[foldIndex] = new ArrayList(2000);
                dcDataFolds[foldIndex] = new ArrayList(2000);
                foldIndexes[foldIndex] = new ArrayList<>(2000);
            }
            for (int foldIndex = 0; foldIndex < data.size(); foldIndex++) {
                choice = randa.nextFloat();
                if (choice < 0.3) {
                    dataFolds[1].add(data.get(foldIndex));
                    dcDataFolds[1].add(data.get(foldIndex));
                    foldIndexes[1].add(foldIndex);
                } else {
                    dataFolds[0].add(data.get(foldIndex));
                    dcDataFolds[0].add(data.get(foldIndex));
                    foldIndexes[0].add(foldIndex);
                }
            }
            // In an unlikely case that one was empty.
            noEmptyFolds = true;
            for (int foldIndex = 0; foldIndex < folds; foldIndex++) {
                if (dataFolds[foldIndex].isEmpty()) {
                    noEmptyFolds = false;
                    break;
                }
            }
        }
        currentTest = foldIndexes[1];
        currentTraining = new ArrayList();
        currentIndexes = new ArrayList();
        currentDCTraining = new ArrayList();
        currentTraining.addAll(dataFolds[0]);
        currentDCTraining.addAll(dcDataFolds[0]);
        currentIndexes.addAll(foldIndexes[0]);
        // Make a classifier congiguration copy.
        classifier = (KNNNB) (copyConfiguration());
        classifier.setDataIndexes(currentIndexes, discDSet);
        ClassificationEstimator currEstimator;
        // Go through the range of neighborhood sizes.
        for (int kCurr = kMin; kCurr <= kMax; kCurr++) {
            classifier.k = kCurr;
            classifier.train();
            // Test the classifier.
            currEstimator = classifier.test(currentTest, discDSet, numClasses);
            if (currEstimator.getAccuracy() > currMaxAcc) {
                currMaxAcc = currEstimator.getAccuracy();
                currMaxK = kCurr;
            }
        }
        k = currMaxK;
    }

    
    /**
     * The default constructor.
     */
    public KNNNB() {
    }

    
    /**
     * Initialization.
     * 
     * @param k Integer that is the neighborhood size.
     */
    public KNNNB(int k) {
        this.k = k;
    }

    
    /**
     * Initialization.
     * 
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public KNNNB(int k, CombinedMetric cmet) {
        this.k = k;
        setCombinedMetric(cmet);
    }

    
    /**
     * Initialization.
     * 
     * @param discDSet DiscretizedDataSet that is the training data.
     */
    public KNNNB(DiscretizedDataSet discDSet) {
        setDataType(discDSet);
        generateClassesFromDataType();
    }

    
    /**
     * Initialization.
     * 
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscreteCategory[] that is the training data.
     */
    public KNNNB(DiscretizedDataSet discDSet, DiscreteCategory[] dataClasses) {
        setDataType(discDSet);
        setClasses(dataClasses);
    }

    
    /**
     * Initialization.
     * 
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscreteCategory[] that is the training data.
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * probability distribution smoothing.
     */
    public KNNNB(DiscretizedDataSet discDSet, DiscreteCategory[] dataClasses,
            float laplaceEstimator) {
        setDataType(discDSet);
        setClasses(dataClasses);
        this.laplaceEstimator = laplaceEstimator;
    }

    
    /**
     * @param laplaceEstimator Float value that is the Laplace estimator for
     * probability distribution smoothing.
     */
    public void setLaplaceEstimator(float laplaceEstimator) {
        this.laplaceEstimator = laplaceEstimator;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    
    /**
     * @param k Integer that is the neighborhood size for the query.
     */
    public void setKNeighbors(int k) {
        this.k = k;
    }

    
    @Override
    public ValidateableInterface copyConfiguration() {
        KNNNB knnbCopy = new KNNNB();
        knnbCopy.setLaplaceEstimator(laplaceEstimator);
        knnbCopy.setDataType(getDataType());
        knnbCopy.setCombinedMetric(getCombinedMetric());
        knnbCopy.setKNeighbors(k);
        return knnbCopy;
    }

    
    @Override
    public void train() throws Exception {
        if (k <= 0) {
            findK(20, 40);
        }
    }

    
    @Override
    public int classify(DiscretizedDataInstance instance) throws Exception {
        DiscretizedDataSet discDSet = getDataType();
        int[] neighborIndexes = NeighborSetFinder.getIndexesOfNeighbors(
                discDSet.getOriginalData(), instance.getOriginalInstance(), k,
                getCombinedMetric());
        // Create a new data context for training Naive Bayes on the query
        // samples.
        DiscretizedDataSet localTrainingSet = discDSet.getSubsample(
                neighborIndexes);
        DiscreteClassifier classifierNB;
        classifierNB = new DNaiveBayes(localTrainingSet);
        ((DNaiveBayes) classifierNB).setLaplaceEstimator(laplaceEstimator);
        classifierNB.train();
        return classifierNB.classify(instance);
    }

    
    @Override
    public float[] classifyProbabilistically(DiscretizedDataInstance instance)
            throws Exception {
        DiscretizedDataSet discDSet = getDataType();
        int[] neighborIndexes = NeighborSetFinder.getIndexesOfNeighbors(
                discDSet.getOriginalData(), instance.getOriginalInstance(), k,
                getCombinedMetric());
        // Create a new data context for training Naive Bayes on the query
        // samples.
        DiscretizedDataSet localTrainingCollection =
                discDSet.getSubsample(neighborIndexes);
        DiscreteClassifier classifierNB;
        classifierNB = new DNaiveBayes(localTrainingCollection);
        ((DNaiveBayes) classifierNB).setLaplaceEstimator(laplaceEstimator);
        classifierNB.train();
        return classifierNB.classifyProbabilistically(instance);
    }
}
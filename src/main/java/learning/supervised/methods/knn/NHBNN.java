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
import learning.supervised.interfaces.AutomaticKFinderInterface;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.evaluation.cv.MultiCrossValidation;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;

/**
 * This class implements the Naive Hubness-Bayesian k-nearest neighbor algorithm
 * that was proposed in the following paper: A Probabilistic Approach to
 * Nearest-Neighbor Classification: Naive Hubness Bayesian kNN by Nenad Tomasev,
 * Milos Radovanovic, Dunja Mladenic and Mirjana Ivanovic, presented at the
 * Conference on Information and Knowledge Management (CIKM) in 2011 in Glasgow.
 * It is a Naive Bayesian interpretation of the k-nearest neighbor rule. Each
 * neighbor occurrence is interpreted as an event, as a new defining feature for
 * the query instance. Unlike in standard Naive Bayes, though - some of these
 * 'features', or rather feature values never occur on the training data and are
 * observed on the test data. This happens for points that are orphans in the
 * kNN graph on the training data and points like this arise frequently in
 * high-dimensional data due to the hubness phenomenon. Even for non-orphan
 * anti-hub points, deriving proper probability conditionals is difficult. On
 * the other hand, most neighbor occurrences in high-dimensional data are hub
 * occurrences and for those points it is possible to derive good
 * class-conditional occurrence probabilities and occurrence-conditional class
 * affiliation probabilities as well. In any case, it is possible to apply the
 * modified Naive Bayes rule for classification based on the kNN set. This
 * approach has later been shown to be quite promising in class-imbalanced
 * high-dimensional data. It was also a basis for development of ANHBNN that was
 * later presented at ECML/PKDD 2013 in Prague.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NHBNN extends Classifier implements AutomaticKFinderInterface,
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {

    private static final long serialVersionUID = 1L;
    // The default anti-hub threshold.
    private int thetaValue = 2;
    // Neighborhood size.
    private int k = 5;
    // The object that holds the kNN information.
    private NeighborSetFinder nsf = null;
    // The training dataset.
    private DataSet trainingData = null;
    // Number of classes in the data.
    private int numClasses = 0;
    // Class-conditional neighbor occurrence frequencies.
    private float[][] classDataKNeighborRelation = null;
    // Prior class distribution.
    private float[] classPriors = null;
    // The smoothing factor.
    private float laplaceEstimator = 0.05f;
    // Neighbor occurrence frequencies on the training data.
    private int[] neighbOccFreqs = null;
    // Several arrays for different types of local and global anti-hub vote
    // estimates, or in this case - conditional probability estimates and
    // approximations.
    private float[][][] localHClassDistribution = null;
    private float[][] classToClassPriors = null;
    // Upper triangular distance matrix.
    private float[][] distMat = null;
    // Variable holding the current estimate type.
    private int localEstimateMethod = GLOBAL;
    // The parameter that governs how much emphasis is put on the actual
    // occurrence counts for anti-hub estimates.
    private float alphaParam = 0.4f;
    // Estimation type constants.
    public static final int GLOBAL = 0;
    public static final int LOCALH = 1;
    private static final int K_LOCAL_APPROXIMATION = 20;
    private boolean noRecalc = false;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("thetaValue", "Anti-hub cut-off point for treating"
                + "anti-hubs as a special case.");
        paramMap.put("localEstimateMethod", "Anti-hub handling strategy.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setConferenceName("Conference on Information and Knowledge "
                + "Management");
        pub.addAuthor(Author.NENAD_TOMASEV);
        pub.addAuthor(Author.MILOS_RADOVANOVIC);
        pub.addAuthor(Author.DUNJA_MLADENIC);
        pub.addAuthor(Author.MIRJANA_IVANOVIC);
        pub.setTitle("A Probabilistic Approach to Nearest-Neighbor "
                + "Classification: Naive Hubness Bayesian kNN");
        pub.setYear(2011);
        pub.setStartPage(183);
        pub.setEndPage(195);
        pub.setPublisher(Publisher.SPRINGER);
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "NHBNN";
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

    @Override
    public void findK(int kMin, int kMax) throws Exception {
        DataSet dset = trainingData;
        float currMaxAcc = -1f;
        int currMaxK = 0;
        int currMaxTheta = 0;
        float maxAlphaParam = 0.4f;
        int bestApproximationMethod = LOCALH;
        NHBNN classifier;
        ArrayList<DataInstance> data = dset.data;
        Random randa = new Random();
        ArrayList[] dataFolds = null;
        ArrayList currentTraining;
        ArrayList[] foldIndexes = null;
        ArrayList<Integer> currentIndexes;
        ArrayList<Integer> currentTest;
        // Generate folds for training and test.
        int folds = 2;
        float choice;
        boolean noEmptyFolds = false;
        while (!noEmptyFolds) {
            dataFolds = new ArrayList[folds];
            foldIndexes = new ArrayList[folds];
            for (int j = 0; j < folds; j++) {
                dataFolds[j] = new ArrayList(2000);
                foldIndexes[j] = new ArrayList<>(2000);
            }
            for (int j = 0; j < data.size(); j++) {
                choice = randa.nextFloat();
                if (choice < 0.15) {
                    dataFolds[1].add(data.get(j));
                    foldIndexes[1].add(j);
                } else {
                    dataFolds[0].add(data.get(j));
                    foldIndexes[0].add(j);
                }
            }
            // Check to see if some have remained empty, though it is highly
            // unlikely, since only 2 folds are used in this implementation.
            noEmptyFolds = true;
            for (int j = 0; j < folds; j++) {
                if (dataFolds[j].isEmpty()) {
                    noEmptyFolds = false;
                    break;
                }
            }
        }
        // Generate training and test datasets from the random folds.
        currentTest = foldIndexes[1];
        currentTraining = new ArrayList();
        currentIndexes = new ArrayList();
        currentTraining.addAll(dataFolds[0]);
        currentIndexes.addAll(foldIndexes[0]);
        classifier = (NHBNN) (copyConfiguration());
        classifier.setDataIndexes(currentIndexes, dset);
        ClassificationEstimator currEstimator;
        DataSet dsetTrain = dset.cloneDefinition();
        dsetTrain.data = currentTraining;
        NeighborSetFinder nsfAux;
        ArrayList<Integer> indexPermutation = MultiCrossValidation.
                getDataIndexes(currentIndexes, dset);
        for (int i = 0; i < dsetTrain.size(); i++) {
            dsetTrain.data.set(i, dset.data.get(indexPermutation.get(i)));
        }
        // Prepare the sub-training kNN sets.
        if (distMat == null) {
            // Calculate the distance matrix from scratch if not already
            // available.
            nsfAux = new NeighborSetFinder(dsetTrain, getCombinedMetric());
  
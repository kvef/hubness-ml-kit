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
package learning.supervised.meta.boosting;

import algref.Author;
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import util.ArrayUtil;
import util.BasicMathUtil;

/**
 * This class implements a classical multi-class boosting method AdaBoost.M2. It
 * re-weights the instances so as to emphasize difficult examples. It also
 * provides a label weighting function for each instance. The final
 * classification is reached by a weighted ensemble vote over all generated
 * models.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class AdaBoostM2 extends Classifier implements
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    // The base classifier to use in boosting.
    private BoostableClassifier weakLearner;
    // All of the trained models.
    private ArrayList<BoostableClassifier> trainedModels;
    private int numIterationsTrain = 100;
    private int numIterationsTest;
    private int numClasses = 1;
    float[] classPriors;
    private DataSet trainingData;
    private Category[] categories;
    // Distributions over the training data.
    private ArrayList<double[]> distributions;
    // Class-conditional difficulty weights.
    private ArrayList<double[][]> weightsPerLabel;
    // Label weighting function.
    private ArrayList<double[][]> labelWeightingFunction;
    // Total instance difficulty weights.
    private ArrayList<double[]> totalWeights;
    // Pseudo-loss over the iterations.
    private ArrayList<Double> pseudoLoss;
    // Loss ratios over the iterations, used to determine learner weights.
    private ArrayList<Double> lossRatios;
    // Learner predictions over the iterations.
    private ArrayList<double[][]> predictions;
    // For corrections when the weights approach the minimal double value too
    // much.
    private static final double SAFE_FACTOR = Math.pow(2, 50);
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("numIterationsTrain", "Number of iterations to use for"
                + "boosting.");
        HashMap<String, String> baseParamMap =
                weakLearner.getParameterNamesAndDescriptions();
        HashMap<String, String> resultingMap = new HashMap<>();
        resultingMap.putAll(paramMap);
        baseParamMap.putAll(baseParamMap);
        return resultingMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("A Decision-Theoretic Generalization of On-Line Learning "
                + "and an Application to Boosting");
        pub.addAuthor(new Author("Yoav", "Freund"));
        pub.addAuthor(new Author("Robert E.", "Schapire"));
        pub.setPublisher(Publisher.ACADEMIC_PRESS);
        pub.setJournalName("Journal of Computer and System Sciences");
        pub.setYear(1997);
        pub.setStartPage(119);
        pub.setEndPage(139);
        pub.setVolume(55);
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    /**
     * Initialization.
     *
     * @param weakLearner BoostableClassifier that is to be boosted.
     */
    public AdaBoostM2(BoostableClassifier weakLearner) {
        this.weakLearner = weakLearner;
    }

    /**
     * Initialization.
     *
     * @param weakLearner BoostableClassifier that is to be boosted.
     * @param numIterations Integer that is the number of boosting iterations.
     */
    public AdaBoostM2(BoostableClassifier weakLearner, int numIterations) {
        this.weakLearner = weakLearner;
        this.numIterationsTrain = numIterations;
    }

    /**
     * Initialization.
     *
     * @param weakLearner BoostableClassifier that is to be boosted.
     * @param numIterations Integer that is the number of boosting iterations.
     * @param trainingData DataSet object to train the model on.
     */
    public AdaBoostM2(BoostableClassifier weakLearner, int numIterations,
            DataSet trainingData) {
        this.weakLearner = weakLearner;
        this.numIterationsTrain = numIterations;
        this.trainingData = trainingData;
    }

    @Override
    public String getName() {
        if (weakLearner != null) {
            return "AdaBoostM2(" + weakLearner.getName() + ")";
        } else {
            return "AdaBoostM2";
        }
    }

    @Override
    public void train() throws Exception {
        NeighborSetFinder nsf = getNSF();
        float[][] dMat = getDistMatrix();
        if (weakLearner instanceof NSFUserInterface && nsf == null) {
            // Then we should calculate the NSF object, for latter speed-ups.
            nsf = new NeighborSetFinder(trainingData, getCombinedMetric());
            if (dMat != null) {
                nsf.setDistances(dMat);
            } else {
                nsf.calculateDistances();
            }
            nsf.calculateNeighborSets(((NSFUserInterface)weakLearner).
                    getNeighborhoodSize());
        }
        int numInstances = trainingData.size();
        classPriors = trainingData.getClassPriors();
        // Because of the interfaces it pays off to duplicate the matrix in a
        // different format.
    
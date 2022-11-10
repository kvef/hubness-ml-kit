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
    // Learn
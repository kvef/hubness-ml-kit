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
    private File inConfigFile, inDir, outDir, currOutDSD
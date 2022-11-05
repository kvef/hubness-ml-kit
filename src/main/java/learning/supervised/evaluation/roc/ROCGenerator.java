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
package learning.supervised.evaluation.roc;

import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.discrete.tranform.EntropyMDLDiscretizer;
import distances.primary.CombinedMetric;
import distances.primary.DistanceMeasure;
import ioformat.FileUtil;
import ioformat.SupervisedLoader;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import learning.supervised.Classifier;
import learning.supervised.ClassifierFactory;
import learning.supervised.DiscreteClassifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import sampling.UniformSampler;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * This script enables ROC classifier evaluation.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ROCGenerator {

    /**
     * This script runs ROC analysis on a list of classifiers for a specified
     * dataset and specified neighborhood size (in case of kNN methods.)
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        // Command line parameter processing.
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inFile", "Path to the input data file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-inClassifierFile", "File containing a comma-separated"
                + "list of the classifiers to evaluate.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Path to the output file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-k", "Neighborhood size.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-metric", "String that is the desired metric.",
                CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        File inFile = new File((String) clp.getParamValues("-inFile").get(0));
        File outFile = new File((String) clp.getParamValues("-outFile").get(0));
        File inClassifierFile = new File((String) clp.getParamValues(
                "-inClassifierFile").get(0));
        int k = (Integer) clp.getParamValues("-k").get(0);
        // Set the chosen metric.
        CombinedMetric cmet = new CombinedMetric();
        Class currFloatMet = Class.forName((String) clp.getParamValues(
                "-metric").get(0));
        cmet.setFloatMetric((DistanceMeasure) (currFloatMet.newInstance()));
        cmet.setCombinationMethod(CombinedMetric.DEFAULT);
        // Data load.
        DataSet dset = SupervisedLoader.loadData(inFile.getPath(), false);
        dset.normalizeFloats();
        // Generate a discretized dataset, if needed.
        DiscretizedDataSet discDset = new DiscretizedDataSet(dset);
        EntropyMDLDiscretizer discretizer = new EntropyMDLDiscretizer();
        discretizer.setDataSet(dset);
        // Count the number of classes in the data.
        int numCategories = dset.countCategories();
        discretizer.setNumCategories(numCategories);
        discretizer.setDiscretizedDataSet(discDset);
        discretizer.discretizeAll();
        discDset.discretizeDataSet(dset);
        int[] trainingIndexes;
        DataSet trainingData;
        DataSet testData;
        DiscretizedDataSet trainingDataDisc;
        DiscretizedDataSet testDataDisc;
        // Get the training and test data splits.
        do {
            trainingIndexes = UniformSampler.getSample(dset.size(),
                    (int) (0.7f * dset.size()));
            trainingData = dset.cloneDefinition();
            testData = dset.cloneDefinition();
            trainingDataDisc = discDset.cloneDefinition();
            trainingDataDisc.setOriginalData(trainingData);
            testDataDisc = discDset.cloneDefinition();
            testDataDisc.setOriginalData(testData);
            trainingData.data = new ArrayList<>(trainingIndexes.length);
            testData.data = new ArrayList<>(dset.size() -
                    trainingIndexes.length);
            trainingDataDisc.data = new ArrayList<>(trainingIndexes.length);
            testDataDisc.data = new ArrayList<>(dset.size()
                    - trainingIndexes.length);
            Arrays.sort(trainingIndexes);
            for (int i = 0; i < trainingIndexes.length - 1; i++) {
                trainingData.addDataInstance(dset.getInstance(
                        trainingIn
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
package learning.supervised.methods;

import algref.Author;
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import combinatorial.Permutation;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import sampling.UniformSampler;
import statistics.HigherMoments;

/**
 * This class implements the robust soft learning vector quantization
 * classification method.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class RSLVQ extends Classifier implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private DataSet trainingData = null;
    private int numClasses = 0;
    // Each class is represented by a number of prototypes.
    private DataInstance[][] classPrototypes;
    private float[][] protoDispersions;
    // Number of prototypes per class.
    private int numProtoPerClass = 5;
    // The learning rates.
    float alphaProto = 0.5f;
    float alphaVariance = 0.3f;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("alphaProto", "Learning rate.");
        paramMap.put("alphaVariance", "Learning rate.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Soft Learning Vector Quantization");
        pub.addAuthor(new Author("Sambu", "Seo"));
        pub.addAuthor(new Author("Klaus", "Obermayer"));
        pub.setJournalName("Neural Computation");
        pub.setYear(2003);
        pub.setStartPage(1589);
        pub.setEndPage(1604);
        pub.setVolume(15);
        pub.setIssue(7);
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "Robust Soft Learning Vector Quantization";
    }
    
    /**
     * Default constructor.
     */
    public RSLVQ() {
    }

    /**
     * Initialization.
     *
     * @param dataClasses Category[] representing the training data.
     */
    public RSLVQ(Category[] dataClasses) {
        setClasses(dataClasses);
        numClasses = dataClasses.length;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object representing the training data.
     * @param numClasses Integer that is the number of classes in the data.
     */
    public RSLVQ(DataSet dset, int numClasses) {
        this.trainingData = dset;
        this.numClasses = numClasses;
        setData(dset.data, dset);
    }

    /**
     * Initialization.
     *
     * @param dataClasses Category[] representing the training data.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public RSLVQ(Category[] dataClasses, CombinedMetric cmet) {
        setClasses(dataClasses);
        numClasses = dataClasses.length;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object representing the training data.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public RSLVQ(DataSet dset, int numClasses, CombinedMetric cmet) {
        this.trainingData = dset;
        this.numClasses = numClasses;
        setData(dset.data, dset);
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param dset DataSet object representing the training data.
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numProtoPerClass Integer that is the number of prototypes per
     * class to use.
     */
    public RSLVQ(DataSet dset, int numClasses, CombinedMetric cmet,
            int numProtoPerClass) {
        this.trainingData = dset;
        this.numClasses = numClasses;
        setData(dset.data, dset);
        setCombinedMetric(cmet);
        this.numProtoPerClass = numProtoPerClass;
    }

    /**
     * Initialization.
     *
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public RSLVQ(int numClasses, CombinedMetric cmet) {
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
    }

    /**
     * Initialization.
     *
     * @param numClasses Integer that is the number of classes in the data.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numProtoPerClass Integer that is the number of prototypes per
     * class to use.
     */
    public RSLVQ(int numClasses, CombinedMetric cmet, int numProtoPerClass) {
        this.numClasses = numClasses;
        setCombinedMetric(cmet);
        this.numProtoPerClass = numProtoPerClass;
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        RSLVQ classifierCopy = new RSLVQ(numClasses, getCombinedMetric());
        return classifierCopy;
    }

    @Override
    public void train() throws Exception {
        CombinedMetric cmet = getCombinedMetric();
        Category[] dataClasses = getClasses();
        int dim = trainingData.getNumFloatAttr();
        // Initialize the prototypes by random point selection.
        numClasses = dataClasses.length;
        numProtoPerClass = Math.min(50, Math.max(5,
                (trainingData.size() / (numClasses * 15))));
        classPrototypes = new DataInstance[numClasses][];
        protoDispersions = new float[numClasses][];
        // Distances to prototypes.
        float[][] protoDists = new float[numClasses][];
        // Exponential distances to prototypes.
        float[][] protoDistExps = new float[numClasses][];
        int tempInt;
        for (int c = 0; c < numClasses; c++) {
            // A class could in principle have fewer members than the desired
            // number of prototypes, so we have to be careful.
            int prLen = Math.max(numProtoPerClass, dataClasses[c].size());
            classPrototypes[c] = new DataInstance[prLen];
            protoDispersions[c] = new float[prLen];
            protoDists[c] = new float[prLen];
            protoDistExps[c] = new float[prLen];
            int[] indexes = UniformSampler.getSample(
                    dataClasses[c].size(), prLen);
            int[] varianceSample = UniformSampler.getSample(
                    dataClasses[c].size(), 20);
            float[] varDists = new float[(varianceSample.length *
                    (varianceSample.length - 1)) / 2];
            tempInt = -1;
            // Calculate the variance from the random sample.
            for (int i = 0; i < varianceSample.length; i++) {
                for (int j = i + 1; j < varianceSample.length; j++) {
                    varDists[++tempInt] = cmet.dist(
                            dataClasses[c].getInstance(varianceSample[i]),
                            dataClasses[c].getInstance(varianceSample[j]));
                }
            }
            float meanVal = HigherMoments.calculateArrayMean(varDists);
            float stDev = HigherMoments.calculateArrayStDev(meanVal,
                    varDists);
            float varianceEstimate = stDev * stDev;
            for (int i = 0; i < prLen; i++) {
                classPrototypes[c][i] = dataClasses[c].getInstance(
                        indexes[i]).copy();
                protoDispersions[c][i] = varianceEstimate;
            }
        }
        // Iterate through the data according to a random permutation.
        int[] indexPermutation = Permutation.obtainRandomPermutation(
                trainingData.size());
        // Current class and current distance.
        int currClass;
        float currDist;
        // Best selection parameters.
        float minDist;
        int minProtoIndex;
        int minClassIndex;

        float sumAll;
        float[] classSums = new float[numClasses];

        float choosingProbTotal;
        float choosingProbInClass;

        float deltaProtoFact;
        float deltaVariance;

        for (int i : indexPermutation) {
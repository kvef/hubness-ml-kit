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
     * @param cme
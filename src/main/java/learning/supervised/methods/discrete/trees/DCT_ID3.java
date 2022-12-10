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
package learning.supervised.methods.discrete.trees;

import algref.Author;
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.util.DataMineConstants;
import feature.evaluation.DiscreteAttributeValueSplitter;
import feature.evaluation.Info;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.supervised.DiscreteCategory;
import learning.supervised.DiscreteClassifier;
import learning.supervised.evaluation.ValidateableInterface;
import util.BasicMathUtil;

/**
 * This class implements the standard ID3 decision tree classification method.
 * It is a basic tree method.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DCT_ID3 extends DiscreteClassifier implements Serializable {
    
    private static final long serialVersionUID = 1L;

    // The root node.
    private DecisionTreeNode root = null;
    // Boolean arrays controlling which features to consider for the splits.
    private boolean[] acceptableFloat;
    private boolean[] acceptableInt;
    private boolean[] acceptableNominal;
    // Number of classes in the data.
    private int numClasses;
    // The overall majority class in the data.
    private int generalMajorityClass;
    // The overall 
    private float[] classPriors;
    private int totalNumAtt;
    private int currDepth = 0;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Induction of Decision Trees");
        pub.addAuthor(new Author("J. R.", "Quinlan"));
        pub.setPublisher(Publisher.MCGRAW_HILL);
        pub.setJournalName("Machine Learning");
        pub.setYear(1986);
        pub.setStartPage(81);
        pub.setEndPage(106);
        pub.setVolume(1);
        pub.setIssue(1);
        return pub;
    }

    @Override
    public String getName() {
        return "ID3";
    }

    
    /**
     * The default constructor.
     */
    public DCT_ID3() {
    }

    
    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data.
     */
    public DCT_ID3(DiscretizedDataSet discDSet) {
        setDataType(discDSet);
        generateClassesFromDataType();
    }

    
    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscreteCategory[] that is the training data.
     */
    public DCT_ID3(DiscretizedDataSet discDSet, DiscreteCategory[] dataClasses) {
        setClasses(dataClasses);
        setDataType(discDSet);
    }
    
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    
    @Override
    public ValidateableInterface copyConfiguration() {
        return new DCT_ID3();
    }

    
    /**
     * This method makes a subtree below the current node.
     *
     * @param node DecisionTreeNode to expand.
     */
    public void makeTree(DecisionTreeNode node) {
        currDepth++;
        node.depth = currDepth;
        node.currInfoValue = 0;
        if (node.indexes != null && node.indexes.size() >= 1) {
            // Calculate the class distribution within the node.
            node.classPriorsLocal = new float[numClasses];
            for (int i = 0; i < node.indexes.size(); i++) {
                node.classPriorsLocal[(node.discDSet.data.get(
                        node.indexes.get(i))).getCategory()]++;
            }
            float localLargestFreq = 0;
            int numNonZeroClasses = 0;
            for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                if (node.classPriorsLocal[cIndex] > 0) {
                    numNonZeroClasses++;
                    // Normalize the prior.
                    node.classPriorsLocal[cIndex] /=
                            (float) node.indexes.size();
                    // Keep track of the locally largest frequency.
                    if (node.classPriorsLocal[cIndex] > localLargestFreq) {
                        node.majorityClass = cIndex;
                        localLargestFreq = node.classPriorsLocal[cIndex];
                    }
                    node.currInfoValue -= node.classPriorsLocal[cIndex] *
                            BasicMathUtil.log2(node.classPriorsLocal[cIndex]);
                }
            }
            if (numNonZeroClasses == 1) {
                // All instances in the node belong to the same class, so there
                // is no need to expand further. The work is done here.
                currDepth--;
                return;
            }
            if (currDepth <= totalNumAtt) {
                // There are more attributes to try.
                Info infoCalculator = new Info(
                        new DiscreteAttributeValueSplitter(node.discDSet),
                        numClasses);
                int[] typeAndIndex = infoCalculator.
                        getTypeAndIndexOfLowestEvaluatedFeature
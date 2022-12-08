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
import algref.BookPublication;
import algref.Publication;
import algref.Publisher;
import data.representation.discrete.DiscretizedDataInstance;
import data.representation.discrete.DiscretizedDataSet;
import data.representation.util.DataMineConstants;
import feature.evaluation.DiscreteAttributeValueSplitter;
import feature.evaluation.Info;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.DiscreteCategory;
import learning.supervised.DiscreteClassifier;
import learning.supervised.evaluation.ValidateableInterface;

/**
 * This class implements the trivial One-rule classifier that learns a one-level
 * decision tree.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DOneRule extends DiscreteClassifier implements Serializable {
    
    private static final long serialVersionUID = 1L;

    private int[] majorityClassesForSplits = null;
    private float[][] classDistributionsForValues = null;
    private int attType = 0;
    private int index = 0;
    private int[] splitValues;
    private HashMap<Integer, float[]> valueCDistMap;
    private HashMap<Integer, Integer> valueClassMap;
    private float[] classPriors;
    private int majorityClassIndex;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        BookPublication pub = new BookPublication();
        pub.setTitle(" Data Mining: Practical Machine Learning Tools and "
                + "Techniques with Java Implementation");
        pub.addAuthor(new Author("Iain H.", "Witten"));
        pub.addAuthor(new Author("Eibe", "Frank"));
        pub.setPublisher(Publisher.MORGAN_KAUFMANN);
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "One-Rule";
    }

    /**
     * The default constructor.
     */
    public DOneRule() {
    }

    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data.
     */
    public DOneRule(DiscretizedDataSet discDSet) {
        setDataType(discDSet);
        generateClassesFromDataType();
    }

    /**
     * Initialization.
     *
     * @param discDSet DiscretizedDataSet that is the training data context.
     * @param dataClasses DiscretizedDataSet that is the training data.
     */
    public DOneRule(DiscretizedDataSet discDSet,
            DiscreteCategory[] dataClasses) {
        setDataType(discDSet);
        setClasses(dataClasses);
    }

    @Override
    public ValidateableInterface copyConfiguration() {
        return new DOneRule();
    }

    
    @Override
    public void train() throws Exception {
        DiscreteCategory[] dataClasses = getClasses();
        DiscretizedDataSet discDSet = getDataType();
        // Obtain the class priors.
        classPriors = discDSet.getClassPriors();
        // Calculate the majority class.
        majorityClassIndex = 0;
        float majorityClassProb = 0;
        for (int cIndex = 0; cIndex < classPriors.length; cIndex++) {
            if (classPriors[cIndex] > majorityClassProb) {
                majorityClassProb = classPriors[cIndex];
                majorityClassIndex = cIndex;
            }
        }
        // Find the best attribute to split on.
        Info infoCalculator = new Info(
                n
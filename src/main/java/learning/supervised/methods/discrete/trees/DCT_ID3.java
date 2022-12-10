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
    public HashMap<String, String> getParameterNamesAndDescr
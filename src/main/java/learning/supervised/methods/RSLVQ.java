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
    // The learning rate
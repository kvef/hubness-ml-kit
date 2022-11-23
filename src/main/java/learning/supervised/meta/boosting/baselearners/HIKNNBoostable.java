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
package learning.supervised.meta.boosting.baselearners;

import algref.Author;
import algref.ConferencePublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import learning.supervised.meta.boosting.BoostableClassifier;
import util.ArrayUtil;
import util.BasicMathUtil;

/**
 * This class implements the HIKNN algorithm that was proposed in the paper
 * titled: "Nearest Neighbor Voting in High Dimensional Data: Learning from Past
 * Occurrences" published in Computer Science and Information Systems in 2011.
 * The algorithm is an extension of h-FNN that gives preference to rare neighbor
 * points and uses some label information. This is an extension that supports
 * instance weights for boosting.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HIKNNBoostable extends BoostableClassifier implements
        DistMatrixUserInterface, NSFUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    private int k = 5;
    // NeighborSetFinder object for kNN calculations.
    private NeighborSetFinder nsf = null;
    private DataSet trainingData = null;
    private int numClasses = 0;
    // The distance weighting parameter.
    private float mValue = 2;
    private double[][] classDataKNeighborRelation = null;
    // Information contained in the neighbors' labels.
    private float[] labelInformationFactor = null;
    // The prior class distribution.
    private float[] classPriors = null;
    private float laplaceEstimator = 0.001f;
    private int[] neighborOccurrenceFreqs = null;
    private double[] neighborOccurrenceFreqsWeighted = null;
    // The distance matrix.
    private float[][] distMat;
    private boolean noRecalc = true;
    // Boosting weights.
    private double[] instanceWeights;
    private double[][] instanceLabelWeights;
    // Boosting mode.
    public static final int B1 = 0;
    public static final int B2 = 1;
    private int boostingMode = B1;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("mValue", "Exponent for distance weighting. Defaults"
                + " to 2.");
        paramMap.put("boostingMode", "Type of re-weighting procedure.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new Confe
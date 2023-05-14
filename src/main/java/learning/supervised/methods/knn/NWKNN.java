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
package learning.supervised.methods.knn;

import algref.Author;
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
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

/**
 * Implements the algorithm described in Neighbor-weighted K-nearest neighbor
 * for unbalanced text corpus by Songbo Tan in Expert Systems with Applications
 * 28 (2005) 667–671. A weighting factor is included in the voting procedure to
 * compensate for class imbalance.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NWKNN extends Classifier implements DistMatrixUserInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    private int k = 5;
    private DataSet trainingData = null;
    private int numClasses = 0;
    private float[][] distMat;
    private float[] classPriors;
    private float[] classWeights;
    private float weightExponent = 0.25f;
    private float mValue = 2;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("weightExponent", "Exponent for class-specific vote "
                + "weights.");
        paramMap.put("mValue", "Exponent for distance weighting. Defaults"
                + " to 2.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Neighbor-weighted K-nearest neighbor for unbalanced text"
                + " corpus");
        pub.addAuthor(new Author("Songbo", "Tan"));
        pub.setPublisher(Publisher.ELSEVIER);
        pub.setJournalName("Expert Systems with Applications");
        pub.setYear(2005);
        pub.setStartPage(667);
        pub.setEndPage(671);
        pub.setVolume(28);
        pub.setIssue(4);
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "NWKNN";
    }

    @Override
    public void setDistMatrix(float[][] distMatrix) {
        this.distMat = distMatrix;
    }

    @Override
    public float[][] getDistMatrix() {
        return distMat;
    }

    /**
     * @return Float value that is the weight exponent.
     */
    public float getWeightExponent() {
        return weightExponent;
    }

    /**
     * @param exponent Float value that is the weight exponent.
     */
    public void setWeightExponent(float exponent) {
        this.weightExponent = exponent;
    }

    /**
     * Default constructor.
     */
    public NWKNN() {
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public NWKNN(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calcul
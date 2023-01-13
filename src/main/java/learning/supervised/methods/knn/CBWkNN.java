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
import algref.ConferencePublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import learning.supervised.Category;
import learning.supervised.Classifier;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistMatrixUserInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;

/**
 * An algorithm described in the paper titled "Class Based Weighted K-Nearest
 * Neighbor over Imbalance Dataset" that was presented at PAKDD 2013 in Gold
 * Coast, Australia by Harshit Dubey and Vikram Pudi the idea is to assign
 * (query and class) - specific weights to instance votes.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CBWkNN extends Classifier implements DistMatrixUserInterface,
        NSFUserInterface, DistToPointsQueryUserInterface,
        NeighborPointsQueryUserInterface, Serializable {
    
    private static final long serialVersionUID = 1L;

    private DataSet trainingData = null;
    private int numClasses = 0;
    // Predicted labels of the training data by kNN via leave-one-out.
    private int[] knnClassifications;
    // Neighbor coefficient that is used to determine vote weights.
    private float[] neighborCoefficient;
    // Upper triangular distance matrix.
    private float[][] distMat = null;
    // Object that holds and calculates the kNN sets.
    private NeighborSetFinder nsf = null;
    // Neighborhood size.
    private int k = 10;
    // k/mValue first neighbors is used to calculate the weighting factors for
    // the query point.
    private int mValue = 2;
    
    /**
     * Default constructor.
     */
    public CBWkNN() {
    }
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("mValue", "Denominator to divide the neighborhood size by"
                + "in order to obtain the number of neighbors for weighting"
                + "calculations for the query point.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setConferenceName("Pacific-Asian Conference on Knowledge Discovery "
                + "and Data Mining");
        pub.addAuthor(new Author("Harshit", "Dubey"));
        pub.addAuthor(new Author("Vikram", "Pudi"));
        pub.setTitle("Class Based Weighted K-Nearest Neighbor over Imbalance "
                + "Dataset");
        pub.setYear(2013);
        pub.setStartPage(305);
        pub.setEndPage(316);
        pub.setPublisher(Publisher.SPRINGER);
        pub.setDoi("10.1007/978-3-642-37456-2_26");
        pub.setUrl("http://link.springer.com/chapter/10.1007%2F978-3-642-37456-"
                + "2_26");
        return pub;
    }

    @Override
    public void noRecalcs() {
    }

    /**
     * @param mValue Integer that is the number of neighbors to use for
     * determining the weighting factor.
     */
    public void setM(int mValue) {
        this.mValue = mValue;
    }

    /**
     * @return Integer that is the number of neighbors to use for determining
     * the weighting factor.
     */
    public int getM() {
        return mValue;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

 
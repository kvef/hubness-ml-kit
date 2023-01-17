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
import learning.supervised.interfaces.AutomaticKFinderInterface;
import learning.supervised.evaluation.ValidateableInterface;
import distances.primary.CombinedMetric;
import data.representation.DataInstance;
import data.representation.DataSet;
import learning.supervised.Category;
import learning.supervised.Classifier;
import data.neighbors.NeighborSetFinder;
import java.io.Serializable;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * This class implements the basic k-nearest neighbor classifier.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KNN extends Classifier implements AutomaticKFinderInterface,
        DistToPointsQueryUserInterface, NeighborPointsQueryUserInterface,
        Serializable {
    
    private static final long serialVersionUID = 1L;

    // The training dataset.
    private DataSet trainingData = null;
    // The number of classes in the data.
    private int numClasses = 0;
    // The neighborhood size.
    private int k = 1;
    // The prior class distribution.
    private float[] classPriors;
    
    /**
     * Default constructor.
     */
    public KNN() {
    }
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Nearest Neighbor Pattern Classification");
        pub.addAuthor(new Author("T. M.", "Cover"));
        pub.addAuthor(new Author("P. E.", "Hart"));
        pub.setPublisher(Publisher.IEEE);
        pub.setJournalName("IEEE Transactions on Information Theory");
        pub.setYear(1967);
        pub.setStartPage(21);
        pub.setEndPage(27);
        pub.setVolume(13);
        pub.setIssue(1);
        return pub;
    }
    
    @Override
    public long getVersion() {
        return serialVersionUID;
    }

    @Override
    public String getName() {
        return "kNN";
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for distance calculations.
     */
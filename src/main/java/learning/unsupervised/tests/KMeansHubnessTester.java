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
package learning.unsupervised.tests;

import algref.Author;
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import data.generators.util.MultiGaussianMixForClusteringTesting;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import static learning.unsupervised.ClusteringAlg.MIN_ITERATIONS;
import learning.unsupervised.initialization.PlusPlusSeeder;

/**
 * Tracks the localization of hubs and medoids in K-means iterations.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KMeansHubnessTester extends ClusteringAlg {

    private static final int MAX_ITER = 40;
    private float smallestError = Float.MAX_VALUE;
    private int[] bestAssociations = null;
    int[] hubnessArray = null;
    PrintWriter hcdWriter = null;
    // Final centroids after the clustering is done.
    private DataInstance[] endCentroids = null;
    // When the change in calculateIterationError falls below a threshold, we
    // declare convergence and end the clustering run.
    private static final double ERROR_THRESHOLD = 0.001;
    ArrayList<Float> hcMinVect = new ArrayList<>(300);
    ArrayList<Float> hcMaxVect = new ArrayList<>(300);
    ArrayList<Float> hcAvgVect = new ArrayList<>(300);
    ArrayList<Float> mAvgVect = new ArrayList<>(300);
    ArrayList<Float> mMinVect = new ArrayList<>(300);
    ArrayList<Float> mMaxVect = new ArrayList<>(300);
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        return paramMap;
    }

    @Override
    public Publication getPublicationInfo() {
        // The publication info is given for the paper that originally used 
        // hubness tracking in K-means iterations. For a reference on K-means 
        // itself, look up the base KMeans class.
        JournalPublication pub = new JournalPublication();
        pub.setTitle("The Role of Hubness in Clustering High-Dimensional Data");
        pub.addAuthor(Author.NENAD_TOMASEV);
        pub.addAuthor(Author.MILOS_RADOVANOVIC);
        pub.addAuthor(Author.DUNJA_MLADENIC);
        pub.addAuthor(Author.MIRJANA_IVANOVIC);
        pub.setPublisher(Publisher.IEEE);
        pub.setJournalName("IEEE Transactions on Knowledge and Data "
                + "Engineering");
        pub.setYear(2014);
        pub.setStartPage(183);
        pub.setEndPage(195);
        pub.setVolume(6634);
        pub.setDoi("10.1109/TKDE.2013.25");
        pub.setUrl("http://ieeexplore.ieee.org/xpl/articleDetails.jsp?"
                + "arnumber=6427743");
        return pub;
    }

    /**
     * @param hubnessArray An integer array of neighbor occurrence frequencies.
     */
    public void setHubness(int[] hubnessArray) {
        this.hubnessArray = hubnessArray;
    }

    /**
     * Calculate the neighbor occurrence frequencies of data points.
     *
     * @param k Neighborhood size to use for kNN sets.
     * @param cmet CombinedMetric object.
     * @throws Exception
     */
    public void calculateHubness(int k, CombinedMetric cmet) throws Exception {
        if (cmet == null) {
            cmet = CombinedMetric.EUCLIDEAN;
        }
        NeighborSetFinder nsf = new NeighborSetFinder(getDataSet(), cmet);
        nsf.calculateDistances();
        nsf.calculateNeighborSets(k);
        hubnessArray = nsf.getNeighborFrequencies();
    }

    /**
     * The default constructor.
     */
    public KMeansHubnessTester() {
    }

    /**
     
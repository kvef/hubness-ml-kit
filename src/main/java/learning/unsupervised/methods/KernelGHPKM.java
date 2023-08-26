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
package learning.unsupervised.methods;

import algref.Author;
import algref.BookChapterPublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.kernel.Kernel;
import distances.kernel.MinKernel;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.initialization.PlusPlusSeeder;
import util.ArrayUtil;

/**
 * A kernelized version of the GHPKM algorithm, described in the following
 * chapter: "Hubness-proportional clustering of High-dimensional Data" by Nenad
 * Tomasev, Milos Radovanovic, Dunja Mladenic and Mirjana Ivanovic, published in
 * "Partitional Clustering Algorithms" book in 2014.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KernelGHPKM extends ClusteringAlg implements
        distances.kernel.KernelMatrixUserInterface,
        learning.supervised.interfaces.DistMatrixUserInterface,
        data.neighbors.KernelNSFUserInterface {

    private double error; // Calculated incrementally.
    float[][] kmat;
    // Kernel matrix - it HAS diagonal entries, as k(x,x) can possibly vary -
    // not in those kernels based on (x-y), but in those based on xy.
    Kernel ker;
    float[] instanceWeights;
    double[] clusterKerFactors;
    private static final double ERROR_THRESHOLD = 0.001;
    public static final int UNSUPERVISED = 0;
    public static final int SUPERVISED = 1;
    private static final int MAX_ITER = 100;
    boolean unsupervisedHubness = true;
    private float[][] distances = null;
    private double[] cumulativeProbabilities = null;
    private float smallestError = Float.MAX_VALUE;
    private int[] bestAssociations = null;
    private int[] hubnessArray = null;
    private int[] clusterHubIndexes;
    private int k = 10;
    private NeighborSetFinder nsf;
    DataInstance[] endCentroids = null;
    public int probabilisticIterations = 20;
    boolean history = false;
    private ArrayList<int[]> historyIndexArrayList;
    private ArrayList<DataInstance[]> historyDIArrayList;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("unsupervisedHubness", "If true, total neighbor occurrence"
                + "frequencies are used for deriving the weights. If false,"
                + "class-conditional occurrences are also taken into account.");
        paramMap.put("ker", "Kernel.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        BookChapterPublication pub = new BookChapterPublication();
        pub.setTitle("Hubness-based Clustering of High-Dimensional Data");
        pub.addAuthor(Author.NENAD_TOMASEV);
        pub.addAuthor(Author.MILOS_RADOVANOVIC);
        pub.addAuthor(Author.DUNJA_MLADENIC);
        pub.addAuthor(Author.MIRJANA_IVANOVIC);
        pub.setPublisher(Publisher.SPRINGER);
        pub.setBookName("Partitional Clustering Algorithms");
        pub.setYear(2014);
        pub.setStartPage(353);
        pub.setEndPage(386);
        pub.setDoi("10.1007/978-3-319-09259-1_11");
        pub.setUrl("http://link.springer.com/chapter/10.1007/"
                + "978-3-319-09259-1_11");
        return pub;
    }

    public KernelGHPKM() {
    }

    /**
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param ker Kerne
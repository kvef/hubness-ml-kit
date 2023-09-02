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
package learning.unsupervised.methods.multithreaded;

import algref.Author;
import algref.ConferencePublication;
import algref.Publication;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import data.structures.KDDataNode;
import data.structures.KDTree;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.ClusteringError;
import learning.unsupervised.initialization.PlusPlusSeeder;

/**
 * A fast K-means implementation described in: Alsabti, Khaled, Sanjay Ranka,
 * and Vineet Singh. "An Efficient K-Means Clustering Algorithm." This is a
 * slightly modified version of the original, that multi-threads when pruning
 * centroid lists.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MTFastKMeansPlusPlus extends ClusteringAlg {

    private static final double ERROR_THRESHOLD = 0.001;
    private Cluster[] clusters = null;
    private float[] clusterSquareSums = null;
    private int[] clusterNumberOfElements = null;
    private float[][] clusterLinearIntSums = null;
    private float[][] clusterLinearFloatSums = null;
    private DataInstance[] endCentroids = null;
    private int numThreads = 8;
    private volatile int threadCount = 1;
    private boolean printOutIteration = false;
    private Object[] elLock;
    private Object[] sqLock;
    private Object[] linintLock;
    private Object[] linfloatLock;
    private Object[] instanceAddLock;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("alpha", "Weight of the descriptors.");
        paramMap.put("beta", "Weight of the color information.");
        paramMap.put("minClusters", "Minimal number of clusters to try.");
        paramMap.put("maxClusters", "Maximal number of clusters to try.");
        paramMap.put("repetitions", "How many times to repeat for each K.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setConferenceName("IPPS/SPDP Workshop on High Performance Data "
                + "Mining");
        pub.addAuthor(new Author("Khaled", "Alsabti"));
        pub.addAuthor(new Author("Sanjay", "Ranka"));
        pub.addAuthor(new Author("Vineet", "Singh"));
        pub.setTitle("An Efficient K-Means Clustering Algorithm");
        pub.setYear(1998);
        return pub;
    }

    public MTFastKMeansPlusPlus() {
    }

    /**
     * Increases the count of elements in the cluster.
     *
     * @param cIndex Integer that is the cluster index.
     * @param size Integer that is the additional size.
     */
    private void increaseNumEl(int cIndex, int size) {
        synchronized (elLock[cIndex]) {
            clusterNumberOfElements[cIndex] += size;
        }
    }

    /**
     * Increases the square sums in the cluster.
     *
     * @param cIndex Integer that is the cluster index.
     * @param increment Float value that is the increment.
     */
    private void increaseSquareSums(int cIndex, float increment) {
        synchronized (sqLock[cIndex]) {
            clusterSquareSums[cIndex] += increment;
        }
    }

    /**
     * Increases the linear float sums in the cluster.
     *
     * @param cIndex Integer that is the cluster index.
     * @param attIndex Integer that is the index of the target feature.
     * @param increment Float value that is the increment.
     */
    private void increaseLinFloatSums(int cIndex, int attIndex,
            float increment) {
        synchronized (linfloatLock[cIndex]) {
            clusterLinearFloatSums[cIndex][attIndex] += increment;
        }
    }

    /**
     * Increases the linear integer sums in the cluster.
     *
     * @param cIndex Integer that is the cluster index.
     * @param attIndex Integer that is the index of the target feature.
     * @param increment Float value that is the increment.
     */
    private void increaseLinIntSums(int cIndex, int attIndex,
            float increment) {
        synchronized (linintLock[cIndex]) {
            clusterLinearIntSums[cIndex][attIndex] += increment;
        }
    }

    /**
     * Adds an instance to a cluster.
     *
     * @param cIndex Integer that is the cluster index.
     * @param index Integer that is the index of the instance to insert.
     */
    private void addInstanceToCluster(int cIndex, int index) {
        synchronized (instanceAddLock[cIndex]) {
            clusters[cIndex].addInstance(index);
        }
    }

    /**
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     */
    public MTFastKMeansPlusPlus(DataSet dset, CombinedMetric cmet,
            int numClusters) {
        setNumClusters(numClusters);
        setCombinedMetric(cmet);
        setDataSet(dset);
    }

    /**
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     * @param numThreads Integer that is the number of threads to use.
     */
    public MTFastKMeansPlusPlus(DataSet dset, CombinedMetric cmet,
            int numClusters, int numThreads) {
        setNumClusters(numClusters);
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.numThreads = numThreads;
    }

    /**
     * @param dset DataSet object.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     * @param numThreads Integer that is the number of threads to use.
     * @param printOutIteration Boolean flag indicating whether to print out an
     * indicator of each completed iteration to the output stream, which can be
     * used for tracking very long clustering runs.
     */
    public MTFastKMeansPlusPlus(DataSet dset, CombinedMetric cmet,
            int numClusters, int numThreads, boolean printOutIteration) {
        setNumClusters(numClusters);
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.numThreads = numThreads;
        this.printOutIteration = printOutIteration;
    }

    /**
     * @param dset DataSet object.
     * @param numClusters A pre-defined number of clusters.
     */
    public MTFastKMeansPlusPlus(DataSet dset, int numClu
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
import algref.ConferencePublication;
import algref.Publication;
import algref.Publisher;
import combinatorial.Permutation;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Stack;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import util.ArrayUtil;
import util.AuxSort;

/**
 * This class implements the well-known density based DBScan algorithm first
 * proposed in the following paper: Martin Ester, Hans-Peter Kriegel, Jörg
 * Sander, Xiaowei Xu (1996). "A density-based algorithm for discovering
 * clusters in large spatial databases with noise"
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DBScan extends ClusteringAlg implements
        learning.supervised.interfaces.DistMatrixUserInterface,
        data.neighbors.NSFUserInterface {

    private float[][] distances = null;
    private int[] bestAssociations = null;
    // k is used to look for the proper epsilon, according to the kdistances.
    private int k = 10;
    private NeighborSetFinder nsf;
    // We keep an array of visited points.
    private boolean[] visited;
    // minPoints is the minimum number of points in a neighborhood for the point
    // not to be considered noise.
    private int minPoints;
    private float epsilonNeighborhoodDist = Float.MAX_VALUE;
    // Noise percentage should be carefully set.
    private float noisePerc = 0.15f;
    
    @Override
    public Publication getPublicationInfo() {
        ConferencePublication pub = new ConferencePublication();
        pub.setConferenceName("KDD");
        pub.addAuthor(new Author("Martin", "Ester"));
        pub.addAuthor(new Author("Hans-Peter", "Kriegel"));
        pub.addAuthor(new Author("Jorg", "Sander"));
        pub.addAuthor(new Author("Xiaowei", "Xu"));
        pub.setTitle("A density-based algorithm for discovering clusters in "
                + "large spatial databases with noise");
        pub.setYear(1996);
        pub.setPublisher(Publisher.ACM);
        return pub;
    }
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        paramMap.put("minPoints", "Minimal number of points in a neighborhood"
                + "so that the point is not considered to be noise.");
        return paramMap;
    }

    /**
     * This method searches for a good parameter configuration. This is achieved
     * by a pre-defined threshold bias where the distances to the k-th nearest
     * neighbor are sorted and then a certain number is discarded as noise. The
     * borderline k-distance is then taken as a limit.
     *
     * @throws Exception
     */
    public void searchForGoodParameters() throws Exception {
        float[][] kdistances = nsf.getKDistances();
        float[] kthdistance = new float[kdistances.length];
        for (int i = 0; i < kdistances.length; i++) {
            kthdistance[i] = kdistances[i][k - 1];
        }
        int[] rearrIndex = AuxSort.sortIndexedValue(kthdistance, true);
        minPoints = k;
        int threshold = (int) (noisePerc * rearrIndex.length);
        epsilonNeighborhoodDist = kthdistance[threshold];
    }

    /**
     * @return Integer that is the minimal number of points a neighborhood can
     * have not to be considered noise.
     */
    public int getMinPoints() {
        return minPoints;
    }

    /**
     * @param minPoints Integer that is the minimal number of points a
     * neighborhood can have not to be considered noise.
     */
    public void setMinPoints(int minPoints) {
        this.minPoints = minPoints;
    }

    /**
     * @return Epsilon-neighborhood diameter.
     */
    public float getEpsilon() {
        return epsilonNeighborhoodDist;
    }

    /**
     * @param epsilon Float that is the epsilon-neighborhood diameter.
     */
    public void setEpsilon(float epsilon) {
        this.epsilonNeighborhoodDist = epsilon;
    }

    public DBScan() {
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Neighborhood size.
     * @param minPoints Minimal number of points in neighborhoods of non-noisy
     * data points.
     * @param epsilon Diameter of the epsilon-neighborhood.
     */
    public DBScan(DataSet dset, CombinedMetric cmet, int k, int minPoints,
            float epsilon) {
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.k = k;
        this.minPoints = minPoints;
        epsilonNeighborhoodDist = epsilon;
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Neighborhood size.
     */
    public DBScan(DataSet dset, CombinedMetric cmet, int k) {
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.k = k;
    }

    /**
     * @param dset DataSet object for clustering.
     * @param k Neighborhood size.
     * @param minPoints Minimal number of points in neighborhoods of non-noisy
     * data points.
     * @param epsilon Diameter of the epsilon-neighborhood.
     */
    public DBScan(DataSet dset, int k, int minPoints, float epsilon) {
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setDataSet(dset);
        this.k = k;
        this.minPoints = minPoints;
        epsilonNeighborhoodDist = epsilon;
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param k Neighborhood size.
     * @param minPoints Minimal number of points in neighborhoods of non-noisy
     * data points.
     * @param epsilon Diameter of the epsilon-neighborhood.
     * @param noisePerc Expected percentage of noise in the data.
     */
    public DBScan(DataSet dset, CombinedMetric cmet, int k, int minPoints,
            float epsilon, float noisePerc) {
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.k = k;
        this.minPoints = minPoints;
        epsilonNeighborhoodDist = epsilon;
        this.noisePerc = noisePerc;
    }

    /**
     * @param dset DataSet object for clustering.
     * @param k Neighborhood size.
     * @param minPoints Minimal number of points in neighborhoods of non-noisy
     * data points.
     * @param epsilon Diameter of the epsilon-neighborhood.
     * @param noisePerc Expected percentage of noise in the data.
     */
    public DBScan(DataSet dset, int k, int minPoints, float epsilon,
            float noisePerc) {
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setDataSet(dset);
        this.k = k;
        this.minPoints = minPoints;
        epsilonNeighborhoodDist = epsilon;
        this.noisePerc = noisePerc;
    }

    @Override
    public void cluster() throws Exception {
        performBasicChecks();
        flagAsActive();
        DataSet dset = getDataSet();
        int size = dset.size();
        visited = new boolean[size];
        Arrays.fill(visited, false);
        int cNum = 0;
        ArrayList<Cluster> clusters = new ArrayList<>(10);
        bestAssociations = new int[size];
        Arrays.fill(bestAssociations, -1);
        if (epsilonNeighborhoodDist == Float.MAX_VALUE) {
            searchForGoodParameters();
        }
        int neighbSize;
        CombinedMetric cmet = getCombinedMetric();
        // Only calculates them if the current NeighborSetFinder object doesn't
        // have them properly calculated.
        calculateNeighborSets(k, cmet);
        int[] perm = Permutation.obtainRandomPermutation(size);
        for (int i = 0; i < size; i++) {
            if (!visited[perm[i]]) {
                visited[perm[i]] = true;
                neighbSize = queryNumNPoints(perm[i], nsf);
                if (neighbSize < minPoints) {
                    bestAssociations[perm[i]] = -1; // Marked as noise.
                } else {
                    cNum++;
                    Cluster clust = new Cluster(dset, size / 10);
                    expandCluster(perm[i], neighbSize, clust, cNum - 1);
                    clusters.add(clust);
                }
            }
        }
        setClusterAssociations(bestAssociations);
    }

    /**
     * Expands the cluster around the considered core point as much as possible.
     *
     * @param index Index of the considered data point.
     * @param neighbSize Neighborhood size.
     * @param currentCluster Current cluster.
     * @param clustIndex Current cluster index.
     */
    private void expandCluster(int index, int neighbSize,
            Cluster currentCluster, int clustIndex) {
        currentCluster.addInstance(index);
        bestAssociations[index] = clustIndex;
        Stack<Integer> potentialStack = new Stack<>();
        int[][] kneighbors = nsf.getKNeighbors();
        for (int i = 0; i < neighbSize; i++) {
            potentialStack.push(kneighbors[index][i]);
        }
        while (!potentialStack.empty()) {
            int i = potentialStack.pop();
            if (!visited[i]) {
                visited[i] = true;
                int nSize2 = queryNumNPoints(i, nsf);
                if (nSize2 >= minPoints) {
                    for (int j = 0; j < nSize2; j++) {
                        potentialStack.push(kneighbors[i][j]);
                    }
                }
            }
            if (bestAssociations[i] == -1) { // Not yet assigned to a cluster.
                // Assign it to the current cluster.
                bestAssociations[i] = clustIndex;
                currentCluster.addInstance(i);
            }
        }
    }

    /**
     * Counts how many neighbor points are at a distance closer than the epsilon
     * neighborhood diameter.
     *
     * @param index Index of the considered data point.
     * @param nsf NeighborSetFinder object.
     * @return An integer count representing the number of data points closer
     * than epsilon.
     */
    private int queryNumNPoints(int index, NeighborSetFinder nsf) {
        int closePointCounter = 0;
        float[][] kdistances = nsf.getKDistances();
        while (closePointCounter < kdistances[index].length
                && kdistances[index][closePointCounter]
                < epsilonNeighborhoodDist) {
            closePointCounter++;
        }
        return closePointCounter;
    }

    /**
     * Counts how many neighbor points are at a distance closer than the epsilon
     * neighborhood diameter.
     *
     * @param index Index of the considered data point.
     * @param nsf NeighborSetFinder object.
     * @param epsilonNeighborhoodTest Epsilon neighborhood diameter to use for
     * the count.
     * @return An integer count representing the number of data points closer
     * than epsilon.
     */
    private int queryNumNPoints(int index, NeighborSetFinder nsf,
            float epsilonNeighborhoodTest) {
        int closePointCounter = 0;
        float[][] kdistances = nsf.getKDistances();
        while (closePointCounter < kdistances[index].length
                && kdistances[index][closePointCounter]
                < epsilonNeighborhoodTest) {
            closePointCounter++;
        }
        return closePointCounter;
    }

    @Override
    public int[] assignPointsToModelClusters(DataSet dsetTest,
            NeighborSetFinder nsfTest) {
        if (dsetTest == null || dsetTest.isEmpty()) {
            return null;
        } else {
            int[] clusterAssociations = new int[dsetTest.size()];
            Arrays.fill(clusterAssociations, -1);
            CombinedMetric cmet = getCombinedMetric();
            Cluster[] clusterConfiguration = getClusters();
            int numClusters = clusterConfiguration.length;
            Cluster[] clusterConfigurationCopy = new Cluster[numClusters];
            DataInstance[] centroids = new DataInstance[numClusters];
            for (int i = 0; i < centroids.length; i++) {
                try {
                    centroids[i] = clusterConfiguration[i].getCentroid();
                } catch (Exception e) {
                }
            }
            for (int i = 0; i < clusterConfigurationCopy.length; i++) {
                clusterConfigurationCopy[i] = clusterConfiguration[i].copy();
            }
            int cNum = 0;
            int neighbSize = 0;
            float minDist;
            float currentDistance;
            float[][] kdistances = nsfTest.getKDistances();
            float[] kthdistance = new float[kdistances.length];
            for (int i = 0; i < kdistances.length; i++) {
                kthdistance[i] = kdistances[i][k - 1];
            }
            int[] rearrIndex;
            try {
                rearrIndex = AuxSort.sortIndexedValue(kthdistance, true);
            } catch (Exception e) {
                return null;
            }
            int threshold = (int) (
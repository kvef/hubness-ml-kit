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
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import java.util.Arrays;
import java.util.HashMap;
import learning.unsupervised.Cluster;
import learning.unsupervised.ClusteringAlg;
import learning.unsupervised.initialization.PlusPlusSeeder;

/**
 * This class implements the Local K-Hubs algorithm that was analyzed in the
 * paper titled "The Role of Hubness in Clustering High-dimensional Data", which
 * was presented at PAKDD in 2011. Hubness is calculated locally here, within
 * the clusters.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LKH extends ClusteringAlg implements
        learning.supervised.interfaces.DistMatrixUserInterface {

    private static final double ERROR_THRESHOLD = 0.001;
    private static final int MAX_ITER = 40;
    private int[] bestAssociations = null;
    private float[][] distances = null;
    DataInstance[] endCentroids;
    private int k = 10;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        paramMap.put("k", "Neighborhood size.");
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
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

    public LKH() {
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     * @param k Neighborhood size.
     */
    public LKH(DataSet dset, CombinedMetric cmet, int numClusters, int k) {
        setNumClusters(numClusters);
        setCombinedMetric(cmet);
        setDataSet(dset);
        this.k = k;
    }

    /**
     * @param dset
     * @param numClusters
     * @param k
     */
    public LKH(DataSet dset, int numClusters, int k) {
        setNumClusters(numClusters);
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setDataSet(dset);
        this.k = k;
    }

    @Override
    public void cluster() throws Exception {
        performBasicChecks();
        flagAsActive();
        DataSet dset = getDataSet();
        CombinedMetric cmet = getCombinedMetric();
        int numClusters = getNumClusters();
        cmet = cmet != null ? cmet : CombinedMetric.EUCLIDEAN;
        boolean trivial = checkIfTrivial();
        if (trivial) {
            return;
        } // Nothing needs to be done in this case.
        int[] clusterAssociations = new int[dset.size()];
        Arrays.fill(clusterAssociations, 0, dset.size(), -1);
        setClusterAssociations(clusterAssociations);
        distances = new float[dset.size()][];
        for (int i = 0; i < distances.length; i++) {
            distances[i] = new float[distances.length - i - 1];
            for (int j = 0; j < distances[i].length; j++) {
                distances[i][j] = -1;
                // Indicating that this distance hasn't been calculated yet.
                // It's a speed up trick so that not all n^2 / 2 distances need
                // to be calculated in order to find hubs.
            }
        }
        DataInstance[] clusterHubs = new DataInstance[numClusters];
        PlusPlusSeeder seeder = new PlusPlusSeeder(numClusters, dset.data,
                cmet);
        int[] clusterHubIndexes = seeder.getCentroidIndexes();
        for (int i = 0; i < numClusters; i++) {
            clusterAssociations[clusterHubIndexes[i]] = i;
            clusterHubs[i] = (dset.data.get(clusterHubIndexes[i]));
        }
        Cluster[] clusters;
        double errorPrevious;
        double errorCurrent = Double.MAX_VALUE;
        setIterationIndex(0);
        boolean noReassignments;
        boolean errorDifferenceSignificant = true;
        int fi, se;
        int closestHub;
        float smallestDistance;
        float currentDistance;
        // It's best if the first assignment is done before and if the
        // assignments are done at the end of the do-while loop, therefore
        // allowing for better calculateIterationError estimates.
        for (int i = 0; i < clusterAssociations.length; i++) {
            closestHub = -1;
            smallestDistance = Float.MAX_VALUE;
            for (int cIndex = 0; cIndex < numClusters; cIndex++) {
                if (clusterHubIndexes[cIndex] > 0) {
                    if (clusterHubIndexes[cIndex] != i) {
                        fi = Math.min(i, clusterHubIndexes[cIndex]);
                        se = Math.max(i, clusterHubIndexes[cIndex]);
                        if (distances[fi][se - fi - 1] <= 0) {
                            distances[fi][se - fi - 1] =
                                    cmet.dist(dset.data.get(fi),
                                    dset.data.get(se));
                        }
                        currentDistance = distances[fi][se - fi - 1];
                    } else {
                        closestHub = cIndex;
                        break;
                    }
                } else {
                    currentDistance = cmet.dist(dset.data.get(i),
                            clusterHubs[cIndex]);
                }
                if (currentDistance < smallestDistance) {
                    smallestDistance = currentDistance;
                    closestHub = cIndex;
                }
            }
            clusterAssociations[i] = closestHub;
        }
        do {
            nextIteration();
            noReassignments = true;
            clusters = getClusters();
           

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
import algref.Publication;
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
 * Clusterer that performs K-medoids. It is a partitional iterative procedure,
 * assigning points to their nearest medoids throughout the iterations. The
 * desired number of clusters needs to be specified in advance.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KMedoidsPlusPlus extends ClusteringAlg {

    // Final medoids after the clustering is done.
    DataInstance[] endMedoids = null;
    // When the change in calculateIterationError falls below a threshold, we
    // declare convergence and end the clustering run.
    private static final double ERROR_THRESHOLD = 0.001;
    
    @Override
    public HashMap<String, String> getParameterNamesAndDescriptions() {
        HashMap<String, String> paramMap = new HashMap<>();
        return paramMap;
    }
    
    @Override
    public Publication getPublicationInfo() {
        Publication pub = new Publication();
        pub.setTitle("Clustering by means of Medoids, in Statistical Data "
                + "Analysis Based on the L1–Norm and Related Methods");
        pub.addAuthor(new Author("L.", "Kaufmann"));
        pub.addAuthor(new Author("P. J.", "Rousseeuw"));
        pub.setYear(1987);
        pub.setStartPage(405);
        pub.setEndPage(416);
        return pub;
    }

    /**
     * Empty constructor.
     */
    public KMedoidsPlusPlus() {
    }

    /**
     * @param dset DataSet object for clustering.
     * @param cmet CombinedMetric object for distance calculations.
     * @param numClusters A pre-defined number of clusters.
     */
    public KMedoidsPlusPlus(DataSet dset, CombinedMetric cmet,
            int numClusters) {
        setDataSet(dset);
        setCombinedMetric(cmet);
        setNumClusters(numClusters);
    }

    /**
     * @param dset DataSet object for clustering.
     * @param numClusters A pre-defined number of clusters.
     */
    public KMedoidsPlusPlus(DataSet dset, int numClusters) {
        setDataSet(dset);
        setCombinedMetric(CombinedMetric.EUCLIDEAN);
        setNumClusters(numClusters);
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
        DataInstance[] medoids = new DataInstance[numClusters];
        Cluster[] clusters;
        PlusPlusSeeder seeder =
                new PlusPlusSeeder(medoids.length, dset.data, cmet);
        int[] centroidIndexes = seeder.getCentroidIndexes();
        for (int cIndex = 0; cIndex < medoids.length; cIndex++) {
            clusterAssociations[centroidIndexes[cIndex]] = cIndex;
            medoids[cIndex] =
                    dset.getInstance(centroidIndexes[cIndex]).copyContent();
        }
        // When there are no reassignments, we can end the clustering.
        boolean noReassignments;
        double errorPrevious;
        double errorCurrent = Double.MAX_VALUE;
        // This is initialized to true for the first iteration to go through.
        boolean errorDifferenceSignificant = true;
        setIterationIndex(0);
        do {
            nextIteration();
            noReassignments = true;
            for (int i = 0; i < dset.size(); i++) {
                float smallestDistance = Float.MAX_VALUE;
                float currentDistance;
                int closestMedoidIndex = -1;
                for (int cIndex = 0; cIndex < medoids.length; cIndex++) {
                    currentDistance = cmet.dist(
                            dset.getInstance(i), medoids[cIndex]);
                    if (currentDistance < smallestDistance) {
                        smallestDistance = currentDistance;
                        closestMedoidIndex = cIndex;
                    }
                }
                if (closestMedoidIndex != clusterAssociations[i]) {
                    // The point has been assigned to a different cluster.
                    clusterAssociations[i] = closestMedoidIndex;
                    noReassignments = false;
                }
            }
            clusters = getClusters();
            for (int cIndex = 0; cIndex < medoids.length; cIndex++) {
                medoids[cIndex] = clusters[cIndex].getMedoid(cmet);
            }
            errorPrevious = errorCurrent;
            errorCurrent = calculateIterationError(medoids);
            if (getIterationIndex() >= MIN_ITERATIONS) {
                if (DataMineConstants.isAcceptableDouble(errorPrevious)
                        && DataMineConstants.isAcceptableDouble(errorCurrent)
                        && (Math.abs(errorCurrent / errorPrevious) - 1f)
                        < ERROR_THRESHOLD) {
                    errorDifferenceSignificant = false;
                } else {
                    errorDifferenceSignificant = true;
                }
            }
        } while (errorDifferenceSignificant && !noReassignments);
        endMedoids = medoids;
        flagAsInactive();
    }

    @Override
    public int[] assignPointsToModelClusters(DataSet dset,
            NeighborSetFinder nsfTest) {
        if (dset == null || dset.isEmpty()) {
            return null;
        } else {
            int[] clusterAssociations = new int[dset.size()];
            if (endMedoids == null) {
                return clusterAssociations;
            }
            float minDist;
            float dist;
            CombinedMetric cmet = getCombinedMetric();
            cmet = cmet != null ? cmet : CombinedMetric.EUCLIDEAN;
            for (int i = 0; i < dset.size(); i++) {
                minDist = Float.MAX_VALUE;
                for (int cIndex = 0; cIndex < endMedoids.length; cIndex++) {
                    dist = Float.MAX_VALUE;
                    try {
                        dist = cmet.dist(
                                endMedoids[cIndex], dset.getInstance(i));
                    } catch (Exception e) {
                    }
                    if (dist < minDist) {
                        clusterAssociations[i] = cIndex;
                        minDist = dist;
                    }
                }
            }
            return clusterAssociations;
        }
    }

    /**
     * Calculates the iteration calculateIterationError for convergence check.
     *
     * @param medoids An array of cluster medoid objects.
     * @return A sum of squared distances from points to medoids.
     * @throws Exception
     */
    private double calculateIterationError(DataInstance[] medoids)
            throws Exception {
        DataSet dset = getDataSet();
        CombinedMetric cmet = getCombinedMetric();
        int[] clusterAssociations = getClusterAssociations();
        double iterationError = 0;
        float medoidDistance;
        for (int i = 0; i < dset.size(); i++) {
            medoidDistance = cmet.dist(medoids[clusterAssociations[i]],
                    dset.getInstance(i));
            iterationError += medoidDistance * medoidDistance;
        }
        return iterationError;
    }
}
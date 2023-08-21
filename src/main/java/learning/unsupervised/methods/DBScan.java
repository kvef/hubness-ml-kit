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
     * @param cm
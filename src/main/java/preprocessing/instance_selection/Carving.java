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
package preprocessing.instance_selection;

import algref.Author;
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.HitMissNetwork;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import learning.supervised.methods.knn.KNN;

/**
 * This class implements the carving instance selection method based on 
 * calculating the HM scores in Hit-Miss networks with subsequent refinement, as 
 * proposed in the paper titled 'Class Conditional Nearest Neighbor and Large 
 * Margin Instance Selection' by E. Marchiori that was published in IEEE 
 * Transactions on Pattern Analysis and Machine Intelligence in 2010. The method 
 * was proposed for 1-NN classification but this implementation makes it 
 * possible to apply the method for kNN classification with k > 1 as well. 
 * Whether that is always appropriate or not remains to be seen, but it gives 
 * the users the option for experimentation. This method severely reduces the 
 * number of examples in practice if kHM == 1 is used and this is definitely 
 * then only good for 1-NN classification. Therefore, it is a good idea to use 
 * larger kHM values, possibly kHM == k for k-NN classification. This is not 
 * strictly enforced, in order to allow for free experimentation, but it is 
 * something that should be kept in mind while experimenting.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Carving extends InstanceSelector implements NSFUserInterface {
    
    public static final int DEFAULT_NEIGHBORHOOD_SIZE = 1;
    public static final int DEFAULT_NUM_THREADS = 8;
    // Object that holds the kNN sets.
    private NeighborSetFinder nsf;
    // The upper triangular distance matrix on the data.
    private float[][] distMat;
    // Hit-Miss network on the data, used for calculating the HM scores.
    private List<HitMissNetwork> hmNetworks;
    // The neighborhood size to use for the hit-miss network.
    private int kHM = DEFAULT_NEIGHBORHOOD_SIZE;
    private int numThreads = DEFAULT_NUM_THREADS;
    private boolean permitNoChangeInclusions = true;
    
    private HMScore internalReducer;
    
    /**
     * Default constructor.
     */
    public Carving() {
    }

    /**
     * Initialization.
     *
     * @param nsf Neighbor set finder object with some existing kNN info.
     * @param kHM Integer representing the neighborhood size to use for the
     * hit-miss network.
     */
    public Carving(NeighborSetFinder nsf, int kHM) {
        this.nsf = nsf;
        if (nsf == null) {
            throw new IllegalArgumentException("Null kNN object provided.");
        }
        setOriginalDataSet(nsf.getDataSet());
        this.distMat = nsf.getDistances();
        this.kHM = kHM;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to reduce.
     * @param distMat float[][] that is the upper triangular distance matrix on
     * the data.
     * @param kHM Integer that is the neighborhood size to use for generating
     * the hit-miss network.
     */
    public Carving(DataSet dset, float[][] distMat, int kHM) {
        setOriginalDataSet(dset);
        this.distMat = distMat;
        this.kHM = kHM;
    }

    /**
     * @param permitNoChangeInclusions Boolean flag indicating whether to
     * consider elements for incremental inclusion when they have no visible
     * negative or positive effect or to stop the process when such an element
     * is reached. If set to false, a very small number of prototypes is
     * selected. If set to true, a much lower error is achieved.
     */
    public void setInclusionPermissions(boolean permitNoChangeInclusions) {
        this.permitNoChangeInclusions = permitNoChangeInclusions;
    }

    /**
     * @param numThreads Integer that is the number of threads to use in parts
     * of the code where multi-threading is supported.
     */
    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }

    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Class Conditional Nearest Neighbor and Large Margin "
                + "Instance Selection");
        pub.addAuthor(new Author("E.", "Marchiori"));
        
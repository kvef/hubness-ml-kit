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
package data.neighbors;

import data.representation.DataSet;
import distances.primary.CombinedMetric;
import ioformat.SupervisedLoader;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import learning.supervised.Category;
import util.BasicMathUtil;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * This class implements the methods for calculating the hit-miss neighbor sets 
 * on the training data in supervised learning. Hits are those neighbor 
 * occurrences where labels match and misses are the neighbor occurrences where 
 * the labels differ. Nearest miss distances can be used to estimate the margins
 * in 1-NN classification and can be used for large-margin NN instance selection
 * and classification.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HitMissNetwork {
    
    public static final int DEFAULT_NEIGHBORHOOD_SIZE = 1;
    // Data to generate the network for.
    private DataSet dset;
    // The upper triangular distance matrix.
    private float[][] dMat;
    // Neighborhood size to use.
    private int k = DEFAULT_NEIGHBORHOOD_SIZE;
    
    // Tabu maps for restricted kNN search within the class or outside the 
    // class.
    private HashMap<Integer, Integer>[] tabuMapsClassQueries;
    private HashMap<Integer, Integer>[] tabuMapsClassComplementQueries;
    
    // The kNN sets for the hits.
    private int[][] knHits;
    private float[] hitNeighbOccFreqs;
    private List<Integer>[] hitReverseNNSets;
    // The kNN sets for the misses.
    private int[][] knMisses;
    private float[] missNeighbOccFreqs;
    private List<Integer>[] missReverseNNSets;
    
    /**
     * Default constructor.
     */
    public HitMissNetwork() {
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to generate the network for.
     * @param dMat float[][] that is the upper triangular distance matrix.
     */
    public HitMissNetwork(DataSet dset, float[][] dMat) {
        this.dset = dset;
        this.dMat = dMat;
    }
    
    /**
     * Initialization.
     * 
     * @param dset DataSet object to generate the network for.
     * @param dMat float[][] that is the upper triangular distance matrix.
     * @param k Integer that is the neighborhood size.
     */
    public HitMissNetwork(DataSet dset, float[][] dMat, int k) {
        this.dset = dset;
        this.dMat = dMat;
        this.k = k;
    }
    
    /**
     * This method generates the hit/miss network.
     * @throws Exception 
     */
    public void generateNetwork() throws Exception {
        // First check for trivial cases.
        if (dset == null || dset.isEmpty()) {
            throw new Exception("No data provided.");
        }
        int dSize = dset.size();
        if (k <= 0 || k > dSize) {
            throw new Exception("Bad neighborhood size provided: " + k);
        }
        if (dMat == null) {
            throw new Exception("No distance matrix provided.");
        }
        int numClasses = dset.countCategories();
        if (numClasses == 1) {
            throw new Exception("Only one class detected in the data. Use "
                    + "standard kNN extraction methods instead.");
        }
        // Initialize class-specific tabu maps for kNN extraction.
        Category[] classes = dset.getClassesArray(numClasses);
        int minClassSize = dSize;
        for (int c = 0; c < numClasses; c++) {
            if (classes[c].size() < minClassSize) {
                minClassSize = classes[c].size();
            }
        }
        if (k > minClassSize) {
            throw new Exception("Specified neighborhood size exceeds minimum "
                    + "class size. Unable to form hit networks for k: " + k);
        }
        tabuMapsClassQueries = new HashMap[dSize];
        tabuMapsClassComplementQueries = new HashMap[dSize];
        for (int c = 0; c < numClasses; c++) {
            tabuMapsClassQueries[c] = new HashMap<>(dSize);
            for (int cOther = 0; cOther < c; cOther++) {
                for (int index: classes[cOther].getIndexes()) {
                    tabuMapsClassQueries[c].put(index, cOther);
                }
            }
            for (int cOther = c + 1; cOther < numClasses; cOther++) {
                for (int index: classes[cOther].getIndexes()) {
                    tabuMapsClassQueries[c].put(index, cOther);
                }
            }
            tabuMapsClassComplementQueries[c] = new HashMap<>(dSize);
            for (int index: classes[c].getIndexes()) {
                tabuMapsClassComplementQueries[c].put(index, c);
            }
        }
        // Further initialization.
        knHits = new int[dSize][k];
        hitNeighbOccFreqs = new float[dSize];
        hitReverseNNSets = new List[dSize];
        for (int i = 0; i < dSize; i++) {
            hitReverseNNSets[i] = new ArrayList<>(k);
        }
        knMisses = new int[dSize][k];
        missNeighbOccFreqs = new float[dSize];
        missReverseNNSets = new List[dSize];
        for (int i = 0; i < dSize; i++) {
            missReverseNNSets[i] = new ArrayList<>(k);
        }
        // Now calculate the hit and miss kNN sets and incrementally update the 
        // occurrence stats.
        for (int i = 0; i < dSize; i++) {
            knHits[i] = NeighborSetFinder.getIndexesOfNeighbors(
                    dMat, i, k, tabuMapsClassQueries[dset.getLabelOf(i)]);
            knMisses[i] = NeighborSetFinder.getIndexesOfNeighbors(
                    dMat, i, k, tabuMapsClassComplementQueries[
                    dset.getLabelOf(i)]);
            for (int kIndex = 0; kIndex < k; kIndex++) {
                hitNeighbOccFreqs[knHits[i][kIndex]]++;
                missNeighbOccFreqs[knMisses[i][kIndex]]++;
                hitReverseNNSets[knHits[i][kIndex]].add(i);
                missReverseNNSets[knMisses[i][kIndex]].add(i);
            }
        }
    }
    
    /**
     * This method 'recycles' some of the existing kNN information.
     * 
     * @param nsf NeighborSetFinder object holding some pre-computed kNN info.
     * @throws Exception 
     */
    public void generateNetworkFromExistingNSF(NeighborSetFinder nsf) 
            throws Exception {
        if (nsf == null) {
            generateNetwork();
            return;
        }
        int kNSF = nsf.getKNeighbors() != null ?
                nsf.getKNeighbors()[0].length : 0;
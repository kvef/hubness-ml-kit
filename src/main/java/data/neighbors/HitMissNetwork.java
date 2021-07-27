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
     
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
package graph.knn;

import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import graph.basic.DMGraph;
import graph.basic.DMGraphEdge;

/**
 * This class models the KNN graph on the specified DataSet for the specified
 * metric. An edge (x,y) exists if y is in a k-NN list of x.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class KNNGraph extends DMGraph {

    private NeighborSetFinder nsf;
    private int[][] kneighbors = null;
    private CombinedMetric cmet;
    private int k = 1;

    /**
     * The basic constructor.
     */
    public KNNGraph() {
    }

    /**
     * Initialization.
     *
     * @param networkName String that is the graph name.
     * @param networkDescription String that is the graph description.
     */
    public KNNGraph(String networkName, String networkDescription) {
        this.networkName = networkName;
        this.networkDescription = networkDescription;
    }

    /**
     * Initialization.
     *
     * @param networkName String that is the graph name.
     * @param networkDescription String that is the graph description.
     * @param vertices DataSet that holds the vertex information. This is where
     * the kNN graph will be derived from.
     * @param cmet CombinedMetric object for distance calculations.
     */
    public KNNGraph(String networkName, String networkDescription,
            DataSet vertices, CombinedMetric cmet) {
        this.networkName = networkName;
        this.networkDescription = networkDescription;
        this.vertices = vertices;
        this.cmet = cmet;
    }

    /**
     * Initialization.
     *
     * @param networkName String that is the graph name.
     * @param networkDescription String that is the graph description.
     * @param vertices DataSet that holds the vertex information. This is where
     * the kNN graph will be derived from.
     * @param kneighbors CombinedMetric object for distance calculations.
     */
    public KNNGraph(String networkName, String networkDescription,
            DataSet vertices, int[][] kneighbors) {
        this.networkName = networkName;
        this.networkDescription = networkDescription;
        this.vertices = vertices;
        this.kneighbors = kneighbors;
        // We get the actual k value from the neighbor set lengths.
        k = kneighbors[0].length;
    }

    /**
     * Initialization.
     *
     * @param networkName String that is the graph name.
     * @param networkDescription String that is the graph description.
     * @param vertices DataSet that holds the vertex information. This is where
     * the kNN graph will be derived from.
     * @param edges DMGraphEdge[] of edges of the graph, in case they have
     * already been calculated.
     */
    public KNNGraph(String networkName, String networkDescription,
            DataSet vertices, DMGraphEdge[] edges) {
        this.networkName = networkName;
        this.networkDescription = networkDescription;
        this.vertices = vertices;
        this.edges = edges;
    }

    /**
     * Initialization.
     *
     * @param k Integer that is the neighborhood size.
     */
    public KNNGraph(int k) {
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param networkName String that is the graph name.
     * @param networkDescription String that is the graph description.
     * @param k Integer that is the neighborhood size.
     */
    public KNNGraph(String networkName, String networkDescription, int k) {
        this.networkName = networkName;
        this.networkDescription = networkDescription;
        this.k = k;
    }

    /**
     * Initialization.
     *
     * @param networkName String that is the graph name.
     * @param networkDescription String that is the graph description.
     * @param vertices DataSet that holds the vertex information. This is where
     * the kNN graph will be derived from.
     * @param k Integer that is the neighborhood size.
     * @param cmet CombinedMetric object for di
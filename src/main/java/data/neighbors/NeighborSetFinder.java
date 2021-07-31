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

import data.neighbors.approximate.AppKNNGraphLanczosBisection;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import util.ArrayUtil;
import util.BasicMathUtil;
import util.SOPLUtil;

/**
 * This class implements the functionality for exact kNN search and kNN graph
 * calculations, in various contexts. It also implements the functionality for
 * calculating the neighbor occurrence frequencies, good and bad occurrences,
 * reverse neighbor sets, reverse and direct neighbor set entropies and other
 * hubness-related measures. Functionally, it implements various hubness-based
 * weighting modes and the class-conditional probabilistic model estimates for
 * hubness-aware classification. It is a simple implementation in that the
 * default kNN search and graph construction methods do not rely on additional
 * spatial indexing. The reason for that, though - is that this library is meant
 * primarily for high-dimensional data analysis, where such indexes have been
 * shown to be of little use - and calculating them takes time. In case of large
 * low-to-medium dimensional datasets where spatial indexing can be very useful,
 * alternative implementations should be used. This one is meant for high-dim
 * data instead. Also, in case of large-scale datasets, approximate kNN
 * extensions are to be preferred.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NeighborSetFinder implements Serializable {

    private static final long serialVersionUID = 1L;
    // Dataset to calculate the k-nearest neighbor sets from.
    private DataSet dset = null;
    // The upper triangular distance matrix, as used throughout the library.
    private float[][] distMatrix = null;
    // CombinedMetric object for distance calculations.
    private CombinedMetric cmet = null;
    // The k-nearest neighbor sets. Each row in the table contains the indexes
    // of the k-nearest neighbors for the g
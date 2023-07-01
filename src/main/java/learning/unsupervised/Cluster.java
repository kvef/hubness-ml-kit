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
package learning.unsupervised;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import data.representation.sparse.BOWInstance;
import data.representation.util.DataInstanceDimComparator;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import ioformat.IOARFF;
import learning.unsupervised.evaluation.EmptyClusterException;
import util.ArrayUtil;
import util.DataSortUtil;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

/**
 * Implements the functionality for representing a data cluster.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Cluster implements Serializable {
    
    private static final long serialVersionUID = 1L;

    // The data context that the indexes point to.
    private DataSet dataContext = null;
    public ArrayList<Integer> indexes = null;

    /**
     * The default constructor.
     * @constructor
     */
    public Cluster() {
    }

    /**
     * @constructor @param dset Dataset the indexes will point to.
     */
    public Cluster(DataSet dset) {
        this.dataContext = dset;
        in
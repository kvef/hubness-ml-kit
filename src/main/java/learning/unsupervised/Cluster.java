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
        indexes = new ArrayList<>(Math.max(dset.size(), 10));
    }

    /**
     * @constructor @param dset Dataset the indexes will point to.
     * @param initSize Initial size of the index array.
     */
    public Cluster(DataSet dset, int initSize) {
        this.dataContext = dset;
        indexes = new ArrayList<>(Math.max(initSize, 10));
    }

    /**
     * @return Indexes of instances within the cluster.
     */
    public ArrayList<Integer> getIndexes() {
        return indexes;
    }

    /**
     * @param indexes Indexes of instances within the cluster.
     */
    public void setIndexes(ArrayList<Integer> indexes) {
        this.indexes = indexes;
    }

    /**
     * @return DataSet that the instances belong to.
     */
    public DataSet getDefinitionDataset() {
        return dataContext;
    }

    /**
     * @param DataSet that the instances belong to.
     */
    public void setDefinitionDataset(DataSet dset) {
        this.dataContext = dset;
    }

    /**
     * @param dset DataSet describing the data from within the cluster.
     * @param data ArrayList of data that is assigned to the current cluster.
     */
    public Cluster(DataSet dset, ArrayList<Integer> indexes) {
        this.dataContext = dset;
        this.indexes = indexes;
    }

    /**
     *
     * @param index Index within the cluster.
     * @return Index within the embedding dataset.
     */
    public int getWithinDataSetIndexOf(int index) {
        if (index < 0 || index > indexes.size()) {
            return -1;
        }
        return indexes.get(index);
    }

    /**
     * @param index Index within the cluster.
     * @param dsIndex Index within dataset.
     */
    public void setWithinDataSetIndexOf(int index, int dsIndex) {
        indexes.set(index, dsIndex);
    }

    /**
     * @return All data instances contained within the cluster.
     */
    public ArrayList<DataInstance> getAllInstances() {
        ArrayList<DataInstance> data =
                new ArrayList<>(indexes.size());
        for (int index : indexes) {
            data.add(dataContext.getInstance(index));
        }
        return data;
    }

    /**
     * @param index Index within the cluster.
     * @return Corresponding DataInstance.
     */
    public DataInstance getInstance(int index) {
        return dataContext.getInstance(indexes.get(index));
    }

    /**
     *
     * @return Cluster that contains the entire dataset.
     */
    public static Cluster fromEntireDataset(DataSet dset) {
        ArrayList<Integer> indexes = new ArrayList<>(dset.size());
        for (int i = 0; i < dset.size(); i++) {
            indexes.add(i);
        }
        return new Cluster(dset, indexes);
    }

    /**
     * @param index Index of the instance to be added to the cluster.
     */
    public void addInstance(int index) {
        indexes.add(index);
    }

    /**
     *
     * @param configuration Cluster configuration.
     * @param dataset Data set.
     * @return Cluster associations for a given clustering configuration.
     */
    public static int[] getAssociationsForClustering(Cluster[] configuration,
            DataSet dataset) {
        int[] clusterAssociations = new int[dataset.size()];
        for (int cIndex = 0; cIndex < configuration.length; cIndex++) {
            Cluster c = configuration[cIndex];
            for (int index : c.indexes) {
                clusterAssociations[index] = cIndex;
            }
        }
        return clusterAssociations;
    }

    /**
     * @param associations Integer array representing cluster associations.
     * @param dset Data set.
     * 
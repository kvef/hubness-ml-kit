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
     * @return The cluster configuration array.
     */
    public static Cluster[] getConfigurationFromAssociations(int[] associations,
            DataSet dset) {
        int numClusters = ArrayUtil.max(associations) + 1;
        Cluster[] clusters = new Cluster[numClusters];
        for (int i = 0; i < numClusters; i++) {
            clusters[i] = new Cluster(dset);
        }
        if ((dset != null) && (associations != null)) {
            for (int i = 0; i < dset.size(); i++) {
                if (associations[i] >= 0) {
                    clusters[associations[i]].addInstance(i);
                }
            }
        }
        return clusters;
    }

    /**
     * @return DataInstance that is the cluster centroid.
     * @throws Exception
     */
    public DataInstance getCentroid() throws Exception {
        if (dataContext == null || dataContext.isEmpty() || isEmpty()) {
            throw new EmptyClusterException();
        }
        if (dataContext instanceof BOWDataSet) {
            return getCentroidSparse();
        } else {
            return getCentroidDense();
        }
    }

    /**
     * @return DataInstance that is the cluster centroid.
     * @throws Exception
     */
    private DataInstance getCentroidSparse() throws Exception {
        BOWInstance centroid = new BOWInstance((BOWDataSet) dataContext);
        // I use sparse vectors with headers usually, so the first element holds
        // the cardinality of the list.
        HashMap<Integer, Float> sparseSums = new HashMap<>(1000);
        HashMap<Integer, Integer> sparseCounts = new HashMap<>(1000);
        int[] integerCounts = new int[dataContext.getNumIntAttr()];
        int[] floatCounts = new int[dataContext.getNumFloatAttr()];
        float[] integerSums = new float[dataContext.getNumIntAttr()];
        float[] floatSums = new float[dataContext.getNumFloatAttr()];
        for (int i = 0; i < size(); i++) {
            BOWInstance instance = (BOWInstance) (getInstance(i));
            HashMap<Integer, Float> indexMap =
                    instance.getWordIndexesHash();
            Set<Integer> keys = indexMap.keySet();
            for (int index : keys) {
                float value = indexMap.get(index);
                if (DataMineConstants.isAcceptableFloat(value)) {
                    if (!sparseCounts.containsKey(index)) {
                        sparseCounts.put(index, 1);
                    } else {
                        sparseCounts.put(index, sparseCounts.get(index) + 1);
                    }
                    if (!sparseSums.containsKey(index)) {
                        sparseSums.put(index, 1f);
                    } else {
                        sparseSums.put(index, sparseSums.get(index) + 1);
                    }
                }
            }
            for (int j = 0; j < dataContext.getNumIntAttr(); j++) {
                if (DataMineConstants.isAcceptableInt(
                        getInstance(i).iAttr[j])) {
                    integerSums[j] += getInstance(i).iAttr[j];
                    integerCounts[j]++;
                }
            }
            for (int j = 0; j < dataContext.getNumFloatAttr(); j++) {
                if (DataMineConstants.isAcceptableFloat(
                        getInstance(i).fAttr[j])) {
                    floatSums[j] += getInstance(i).fAttr[j];
                    floatCounts[j]++;
                }
            }
        }
        if (dataContext.getNumNominalAttr() > 0) {
            for (int i = 0; i < dataContext.getNumNominalAttr(); i++) {
                centroid.sAttr[i] = "dummy" + i;
            }
        }
        for (int i = 0; i < d
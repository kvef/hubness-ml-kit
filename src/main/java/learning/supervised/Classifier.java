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
package learning.supervised;

import algref.Citable;
import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import data.representation.sparse.BOWInstance;
import distances.primary.CombinedMetric;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.evaluation.ValidateableInterface;
import learning.supervised.interfaces.DistToPointsQueryUserInterface;
import learning.supervised.interfaces.NeighborPointsQueryUserInterface;
import preprocessing.instance_selection.InstanceSelector;
import util.ArrayUtil;

/**
 * This class implements the methods used for classifier training and testing.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class Classifier implements ValidateableInterface,
        Serializable, Citable {

    private static final long serialVersionUID = 1L;
    private Category[] trainingClasses = null;
    private CombinedMetric cmet = null;

    @Override
    public abstract ValidateableInterface copyConfiguration();

    @Override
    public void setDataIndexes(ArrayList<Integer> currentIndexes,
            Object dataType) {
        if (currentIndexes != null && currentIndexes.size() > 0) {
            if (dataType instanceof BOWDataSet) {
                BOWDataSet bowDSet = (BOWDataSet) dataType;
                ArrayList<BOWInstance> trueDataVect =
                        new ArrayList<>(currentIndexes.size());
                for (int i = 0; i < currentIndexes.size(); i++) {
                    trueDataVect.add((BOWInstance) (bowDSet.data.get(
                            currentIndexes.get(i))));
                }
                setData(trueDataVect, dataType);
            } else if (dataType instanceof DataSet) {
                DataSet dset = (DataSet) dataType;
                ArrayList<DataInstance> trueDataVect =
                        new ArrayList<>(currentIndexes.size());
                for (int i = 0; i < currentIndexes.size(); i++) {
                    trueDataVect.add(dset.data.get(currentIndexes.get(i)));
                }
                setData(trueDataVect, dataType);
            }
        }
    }

    @Override
    public void setData(ArrayList data, Object dataType) {
        if (data != null && !data.isEmpty()) {
            Category[] catArray = null;
            int numClasses = 0;
            int currClass;
            if (data.get(0) instanceof BOWInstance) {
                BOWInstance instance;
                BOWDataSet bowDSet = (BOWDataSet) dataType;
                BOWDataSet bowDSetCopy = bowDSet.cloneDefinition();
                bowDSetCopy.data = data;
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (BOWInstance) (data.get(i));
                        currClass = instance.getCategory();
                        if (currClass > numClasses) {
                            numClasses = currClass;
                        }
                    }
                }
                numClasses = numClasses + 1;
                catArray = new Category[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    catArray[cIndex] = new Category("number" + cIndex, 200,
                            bowDSet);
                    catArray[cIndex].setDefinitionDataset(bowDSetCopy);
                }
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (BOWInstance) (data.get(i));
                        currClass = instance.getCategory();
                        catArray[currClass].addInstance(i);
                    }
                }
            } else if (data.get(0) instanceof DataInstance) {
                DataInstance instance;
                DataSet dset = (DataSet) dataType;
                DataSet dsetCopy = dset.cloneDefinition();
                dsetCopy.data = data;
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        if (currClass > numClasses) {
                            numClasses = currClass;
                        }
                    }
                }
                numClasses = numClasses + 1;
                catArray = new Category[numClasses];
                for (int cIndex = 0; cIndex < numClasses; cIndex++) {
                    catArray[cIndex] = new Category("number" + cIndex, 200,
                            dset);
                    catArray[cIndex].setDefinitionDataset(dsetCopy);
                }
                for (int i = 0; i < data.size(); i++) {
                    if (data.get(i) != null) {
                        instance = (DataInstance) (data.get(i));
                        currClass = instance.getCategory();
                        catArray[currClass].addInstance(i);
                    }
                }
            }
            setClasses(catArray);
        }
    }

    @Override
    public void setClasses(Object[] dataClasses) {
        if (dataClasses == null || dataClasses.length == 0) {
            return;
        }
        Category[] catArray = new Category[dataClasses.length];
        for (int cIndex = 0; cIndex < dataClasses.length; cIndex++) {
            catArray[cIndex] = (Category) (dataClasses[cIndex]);
        }
        setClasses(catArray);
    }

    /**
     * @param dataClasses Category[] representing the training data.
     */
    public void setClasses(Category[] dataClasses) {
        this.trainingClasses = dataClasses;
    }

    /**
     * @return Category[] representing the training data.
     */
    public Category[] getClasses() {
        return trainingClasses;
    }

    /**
     * @param cmet CombinedMetric object for distance calculations.
     */
    public void setCombinedMetric(CombinedMetric cmet) {
        this.cmet = cmet;
    }

    /**
     * @return CombinedMetric object for distance calculations.
     */
    public CombinedMetric getCombinedMetric() {
        return cmet;
    }

    /**
     * This method runs the classifier training.
     */
    public void run() {
        try {
            train();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }

    @Override
    public abstract void train() throws Exception;

    @Override
    public void trainOnReducedData(InstanceSelector reducer) throws Exception {
        train();
    }

    public abstract int classify(DataInstance instance) throws Exception;

    public abstract float[] classifyProbabilistically(DataInstance instance)
            throws Exception;

    /**
     * This method performs batch classification of an array of DataInstance
     * objects.
     *
     * @param instances DataInstance[] array to classify.
     * @return int[] that are the resulting predicted class affiliations.
     * @throws Exception
     */
    public int[] classify(DataInstance[] instances) throws Exception {
        int[] classificationResults;
        if ((instances == null) || (instances.length == 0)) {
            return null;
        } else {
            classificationResults = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                classificationResults[i] = classify(instances[i]);
            }
            return classificationResults;
        }
    }

    /**
     * This method performs batch classification of an array of DataInstance
     * objects.
     *
     * @param instances DataInstance[] array to classify.
     * @return float[][] that are the resulting predicted probabilistic class
     * assignments.
     * @throws Exception
     */
    public float[][] classifyProbabilistically(DataInstance[] instances)
            throws Exception {
        float[][] classificationResults;
        if ((instances == null) || (instances.length == 0)) {
            return null;
        } else {
            classificationResults = new float[instances.length][];
            for (int i = 0; i < instances.length; i++) {
                classificationResults[i] = classifyProbabilistically(
                        instances[i]);
            }
            return classificationResults;
        }
    }

    /**
     * This method performs batch classification of a list of DataInstance
     * objects.
     *
     * @param instances ArrayList<DataInstance> to classify.
     * @return int[] that are the resulting predicted class affiliations.
     * @throws Exception
     */
    public int[] classify(ArrayList<DataInstance> instances) throws Exception {
        int[] classificationResults;
        if ((instances == null) || (instances.isEmpty())) {
            return null;
        } else {
            classificationResults = new int[instances.size()];
            for (int i = 0; i < instances.size(); i++) {
                classificationResults[i] = classify(instances.get(i));
            }
            return classificationResults;
        }
    }

    /**
     * This method performs batch classification of a list of DataInstance
     * objects.
     *
     * @param instances ArrayList<DataInstance> to classify.
     * @return float[][] that are the resulting predicted probabilistic class
     * assignments.
     * @throws Exception
     */
    publ
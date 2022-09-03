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
              
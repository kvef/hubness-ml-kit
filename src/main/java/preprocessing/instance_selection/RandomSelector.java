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

import algref.Publication;
import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import learning.supervised.Category;
import sampling.UniformSampler;

/**
 * Random instance selection. This class implements the NSFUserInterface, which
 * might not seem logical at first - but the idea is to re-use the existing
 * kNN sets for unbiased neighbor occurrence modeling on the training data, when
 * kNN classifiers are present. When they are not, though - this might slow
 * things down a bit. However, there is also a technical catch. In general, it
 * is possible to run batch classifier testing in Hub Miner without actually
 * having the features and the metric implemented - by making dummy data files
 * and loading the pre-computed distance matrix. If the experimental framework
 * is run in that mode and unbiased hubness estimates are required, they are
 * impossible to calculate unless either the NeighborSetFinder object is
 * provided or the distance matrix itself, since the CombinedMetric objects are
 * either dummies themselves or they have no data to re-calculate the distances
 * from.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class RandomSelector extends InstanceSelector
        implements NSFUserInterface {
    
    // To use for calculating the unbiased hubness estimates.
    private NeighborSetFinder nsf;
    
    @Override
    public Publication getPublicationInfo() {
        // This is just stratified random selection. No publications associated 
        // with it.
        return new Publication();
    }

    /**
     * The default constructor.
     */
    public RandomSelector() {
    }

    /**
     * @param originalDataSet
     */
    public RandomSelector(DataSet originalDataSet) {
        setOriginalDataSet(originalDataSet);
    }
    
    @Override
    public void noRecalcs() {
    }

    @Override
    public void setNSF(NeighborSetFinder nsf) {
        this.nsf = nsf;
    }

    @Override
    public NeighborSetFinder getNSF() {
        return nsf;
    }
    
    /**
     * @param nsf NeighborSetFinder object.
     */
    public RandomSelector(NeighborSetFinder nsf) {
        setOriginalDataSet(nsf.getDataSet());
        this.nsf = nsf;
    }

    @Override
    public InstanceSel
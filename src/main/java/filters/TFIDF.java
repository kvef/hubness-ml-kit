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
package filters;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import data.representation.sparse.BOWInstance;
import data.representation.util.DataMineConstants;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;
import util.BasicMathUtil;

/**
 * This class implements the TF-IDF filter that is widely used in text and image
 * analysis for assigning higher weights to more locally frequent terms and
 * features.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class TFIDF implements FilterInterface {

    boolean[] termFeatures;
    int featureType;
    // Boolean flag indicating whether the termFeatures array refers to the
    // sparse features or not. Default is the dense feature mode.
    boolean sparse = false;

    /**
     * Initialization.
     *
     * @param termFeatures A boolean array indicating which features to consider
     * for weighting.
     * @param featureType Integer indicating the feature type of the features.
     */
    public TFIDF(boolean[] termFeatures, int featureType) {
        this.termFeatures = termFeatures;
        this.featureType = featureType;
    }

    @Override
    public DataSet filterAndCopy(DataSet dset) {
        DataSet result = null;
        try {
            result = dset.copy();
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        filter(result);
        return result;
    }

    /**
     * Sets the sparse/dense feature mode.
     *
     * @param sparse True if the sparse features are the target, false if they
     * are not.
     */
    public void setSparse(boolean sparse) {
        this.sparse = sparse;
    }

    /**
     * Performs TF-IDF weighting on float features, dense or sparse.
     *
     * @param dset DataSet to filter.
     */
    public static void filterFloats(DataSet dset) {
        if (ds
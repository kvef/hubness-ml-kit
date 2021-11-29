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
package data.representation.sparse;

import data.representation.DataInstance;
import data.representation.DataSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;

/**
 * This class implements a data holder for sparse data like documents, in a bag
 * of words format.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BOWInstance extends DataInstance {

    public static final int INIT_HASH_SIZE = 500;
    // This map stores the mapping between word indexes from the vocabulary
    // and their weights or counts in the current document.
    private HashMap<Integer, Float> wordIndexHash =
            new HashMap<>(INIT_HASH_SIZE);
    // The data context variable here is named corpus.
    public BOWDataSet corpus;
    // Name or path of the document from which the data was extracted, if
    // available.
    public String documentName;

    public BOWInstance() {
    }

    /**
     * @param corpus BOWDataSet corpus that this BoW belongs to.
     */
    public BOWInstance(BOWDataSet corpus) {
        super(corpus);
        this.corpus = corpus;
    }

    /**
     * Add two sparse representations, sum up the word counts.
     *
     * @param firstBow First bow representation.
     * @param secondBow Second bow representation.
     * @return BOWInstance that is the sum of the two.
   
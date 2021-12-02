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
     */
    public static BOWInstance add(BOWInstance firstBow, BOWInstance secondBow) {
        ArrayList<BOWInstance> pairList = new ArrayList<>(2);
        pairList.add(firstBow);
        pairList.add(secondBow);
        return sumSparse(pairList);
    }

    /**
     * Sums up a list of bow representations by summing up the word counts.
     *
     * @param instances ArrayList<BOWInstance> of sparse representations.
     * @return BOWInstance that is the sum of the list.
     */
    public static BOWInstance sumSparse(ArrayList<BOWInstance> instances) {
        if (instances == null || instances.isEmpty()) {
            return null;
        }
        BOWInstance result = new BOWInstance(instances.get(0).corpus);
        // Iterate through the maps.
        for (BOWInstance instance : instances) {
            HashMap<Integer, Float> instanceIndexHash =
                    instance.getWordIndexesHash();
            Set<Integer> keys = instanceIndexHash.keySet();
            for (int index : keys) {
                if (!result.getWordIndexesHash().containsKey(index)) {
                    result.getWordIndexesHash().put(index,
                            instanceIndexHash.get(index));
                } else {
                    result.getWordIndexesHash().put(index,
                            result.getWordIndexesHash().get(index)
                            + instanceIndexHash.get(index));
                }
            }
        }
        return result;
    }

    /**
     * Multiply the word counts by the scalar value.
     *
     * @param scalarValue Float value to be used in multiplication.
     */
    public void multiplyByScalar(float scalarValue) {
        Set<Integer> keys = wordIndexHash.keySet();
        for (int index : keys) {
            wordIndexHash.put(index, wordIndexHash.get(index) * scalarValue);
        }
    }

    /**
     * Take an average of a list of sparse representations.
     *
     * @param instances ArrayList<BOWInstance> of sparse representations.
     * @return BOWInstance that is the average of the list.
     */
    public static BOWInstance averageSparse(ArrayList<BOWInstance> instances) {
        if (instances == null || instances.isEmpty()) {
            return null;
        }
        BOWInstance result = sumSparse(instances);
        result.multiplyByScalar(1f / ((float) instances.size()));
        return result;
    }

    /**
     * @return The number of different words this representation encodes.
     */
    public int getNumberOfDifferentWords() {
        return wordIndexHash.isEmpty() ? 0 : wordIndexHash.keySet().size();
    }

    @Override
    public DataSet getEmbeddingDataset() {
        return corpus;
    }

    @Override
    public void embedInDataset(DataSet dset) {
        corpus = (BOWDataSet) dset;
    }

    /**
     * @param documentName String that is the document name.
     */
    public void setDocumentName(String documentName) {
        this.documentName = documentName;
    }

    /**
     * @return String that is the document name.
     */
    public String getDocumentName() {
        return documentName;
    }

    /**
     * @return Float that is the sum of all the word frequencies.
     */
    public float getDocumentLength() {
        float length = 0;
        Set<Integer> keys = wordIndexHash.keySet();
        for (int index : keys) {
            length += wordIndexHash.get(index);
        }
        return length;
    }

    /**
     * Simply adds another occurrence of the index to the internal map.
     *
     * @param index Index of the word to add, from the corpus vocabulary.
     */
    public 
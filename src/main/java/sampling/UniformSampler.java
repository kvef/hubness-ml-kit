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
package sampling;

import data.representation.DataInstance;
import data.representation.DataSet;
import java.util.Arrays;
import java.util.Random;
import util.AuxSort;
import util.DataSetJoiner;

/**
 * A class that implements uniform data sampling.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class UniformSampler extends Sampler {

    /**
     * @param repetitions Boolean flag denoting whether to use repetitive
     * sampling or not.
     */
    public UniformSampler(boolean repetitions) {
        super(repetitions);
    }

    /**
     * Gets the indexes of a uniformly spread sample.
     *
     * @param popSize The size of the data.
     * @param sampleSize The size of the sample.
     * @return Integer array of indexes of a uniform sample.
     * @throws Exception
     */
    public static int[] getSample(int popSize, int sampleSize)
            throws Exception {
        float[] rVals = new float[popSize];
        Random randa = new Random();
        for (int i = 0; i < popSize; i++) {
            rVals[i] = randa.nextFloat();
        }
        int[] indexes = AuxSort.sortIndexedValue(rVals, true);
        int[] result = Arrays.copyOf(indexes, Math.min(sampleSize, popSize));
        return result;
    }

    @Override
    public DataSet getSample(DataSet dset, int sampleSize) throws Exception {
        if (sampleSize <= 0 || dset == null || dset.isEmpty()) {
            return new DataSet();
        }
        DataSet sample = new DataSet(
                dset.iAttrNames,
                dset.fAttrNames,
                dset.sAttrNames, sampleSize);
        Random randa = new Random();
        if ((!getRepetitions()) && sampleSize <= dset.size()) {
            // In the case where there are no repetitions, we can essentially
            // just assign a uniform random value to each array member and then
            // sort and take the first sampleSize positions.
            int[] selIndexes = getSample(dset.size(), sampleSize);
            for (int index : selIndexes) {
                DataInstance instanceCopy = dset.data.get(index).copy();
                sample.addDataInstance(instanceCopy);
                instanceCopy.embedInDataset(sample);
            }
        } else {
            // If repetitions are allowed, we just go and sample iteratively.
            int candidate;
            for (int i = 0; i < sampleSize; i++) {
                candidate = randa.nextInt(dset.size());
                sample.addDataInstance(dset.data.get(candidate).copy());
                sample.data.get(i).embedInDataset(sample);
            }
        }
        return sample;
  
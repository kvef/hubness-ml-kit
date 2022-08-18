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
package images.mining.codebook;

import data.representation.images.quantized.QuantizedImageDistribution;
import data.representation.images.quantized.QuantizedImageDistributionDataSet;
import data.representation.images.quantized.QuantizedImageHistogram;
import data.representation.images.quantized.QuantizedImageHistogramDataSet;
import data.representation.images.sift.LFeatRepresentation;
import data.representation.images.sift.LFeatVector;
import distances.primary.LocalImageFeatureMetric;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * SIFT features codebook class for feature quantization.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SIFTCodeBook {

    public static final int DEFAULT_SIZE = 400;
    // Feature vectors that define the codeboook.
    private ArrayList<LFeatVector> codebook = new ArrayList<>();

    /**
     * @return Integer that is the codebook size.
     */
    public int getSize() {
        return codebook.size();
    }

    /**
     * Adds a new vector to the existing codebook.
     *
     * @param v SIFT feature vector to add to the codebook.
     */
    public void addVectorToCodeBook(LFeatVector v) {
        codebook.add(v);
    }

    /**
     * Sets the codebook.
     *
     * @param codebook ArrayList of SIFT feature vectors comprising the current
     * codebook.
     */
    public void setCodeBookSet(ArrayList<LFeatVector> codebook) {
        this.codebook = codebook;
    }

    /**
     * Generates a new QuantizedImageHistogramDataSet context corresponding to
     * this codebook representation.
     *
     * @return A new QuantizedImageHistogramDataSet context corresponding to
     * this codebook representation.
     */
    public QuantizedImageHistogramDataSet getNewHistogramContext() {
        if (codebook != null) {
            return new QuantizedImageHistogramDataSet(codebook.size());
        } else {
            return new QuantizedImageHistogramDataSet(DEFAULT_SIZE);
        }
    }

    /**
     * Generates a new QuantizedImageDistributionDataSet context corresponding
     * to this codebook representation.
     *
     * @return A new QuantizedImageDistributionDataSet context corresponding to
     * this codebook representation.
     */
    public QuantizedImageDistributionDataSet getNewDistributionContext() {
        if (codebook != null) {
            return new QuantizedImageDistributionDataSet(codebook.size());
        } else {
            return new QuantizedImageDistributionDataSet(DEFAULT_SIZE);
        }
    }

    /**
     * Generates a quantized image representation.
     *
     * @param rep SIFTRepresentation of the image to quantize.
     * @param qihDSet QuantizedImageHistogramDataSet data context to use for
     * initializing the quantized representation.
     * @return QuantizedImageHistogram that is the quantized image
     * representation.
     * @throws Exception
     */
    public QuantizedImageHistogram getHistogramForImageRepresentation(
            LFeatRepresentation rep,
            QuantizedImageHistogramDataSet qihDSet) throws Exception {
        QuantizedImageHistogram qih = new QuantizedImageHistogram(qihDSet);
        if (rep == null || rep.isEmpty()) {
            return qih;
        }
        qih.setPath(rep.getPath());
        for (int i = 0; i < rep.data
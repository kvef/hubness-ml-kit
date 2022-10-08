
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
package learning.supervised.evaluation;

import data.representation.util.DataMineConstants;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

/**
 * This class implements the methods for classification prediction quality
 * estimation in terms of accuracy, precision, recall, f1-score and the Matthews
 * correlation coefficient. It calculates all these measures based on the
 * confusion matrix. In the confusion matrix, the element i,j denotes the
 * (average) number of points that were classified as i despite being of class
 * j.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClassificationEstimator {

    private float[][] confusionMatrix = null;
    private float accuracy = 0f;
    private float[] precision = null;
    private float[] recall = null;
    private float avgPrecision = 0f;
    private float avgRecall = 0f;
    // Micro-averaged.
    private float fMeasureMicroAveraged = 0f;
    // Macro-averaged.
    private float fMeasureMacroAveraged = 0f;
    // The Matthews correlation coefficient is used in binary classification
    // tasks.
    private float matthewsCorrelation = 0f;

    /**
     * Initialization.
     *
     * @param confusionMatrix float[][] that is the confusion matrix.
     */
    public ClassificationEstimator(float[][] confusionMatrix) {
        this.confusionMatrix = confusionMatrix;
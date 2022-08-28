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
package ioformat.results.latex;

import data.representation.util.DataMineConstants;
import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import util.CommandLineParser;

/**
 * This class implements a LaTeX table generator for instance selection
 * experiments. There is a biased and non-biased option contain within, which
 * has to do with whether biased or non-biased hubness-aware learning was done
 * on top of instance selections. For more details, see the appropriate papers.
 * This script (as well as other LaTeX generating scripts) were written with
 * very specific result visualizations in mind and can/should be extended and
 * modified to cover other cases that other people might encounted while using
 * this library.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class InstanceSelectionLatexTableSummarizer {

    // Various files that will be examined.
    private File parentDir, selectionMethodsFile, datasetListFile,
            classifierFile, outputFile;
    // Neighborhood size.
    private int k;
    // Arrays of names of instance selection methods, datasets and classifiers.
    private String[] selectionMethods;
    private String[] datasetList;
    private String[] classifiers;
    // Accuracies and standard deviations, for the biased and non-biased case.
    private float[][][] accTableBiased = null;
    private float[][][] accTableUnbiased = null;
    private float[][][] stDevTableBiased = null;
    private float[][][] stDevTableUnbiased = null;
    private float[][] avgAccBiased = null;
    private float[][] avgAccUnbiased = null;
    private boolean[][][] boldUnbiased = null;

    /**
     * Initialization
     *
     * @param parentDir Directory that contains all the experiment
     * sub-directories.
     * @param selectionMethodsFile File that contains a list of the employed
     * instance selection methods.
     * @param datasetListFile File that contains a list of dataset names.
     * @param classifierFile File that contains a list of classifier names.
     * @param k Integer that is the neighborhood size used in the tests.
     * @param outputFile File that will be used to print the generated LaTeX
     * table to.
     */
    public InstanceSelectionLatexTableSummarizer(File parentDir,
            File selectionMethodsFile,
            File datasetListFile, File classifierFile, int k, File outputFile) {
        this.parentDir = parentDir;
        this.selectionMethodsFile = selectionMethodsFile;
        this.datasetListFile = datasetListFile;
        this.classifierFile = classifierFile;
        this.k = k;
        this.outputFile = outputFile;
    }


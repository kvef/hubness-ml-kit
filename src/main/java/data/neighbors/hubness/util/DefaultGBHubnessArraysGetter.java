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
package data.neighbors.hubness.util;

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import distances.secondary.snd.SharedNeighborCalculator;
import distances.sparse.SparseCombinedMetric;
import distances.sparse.SparseCosineMetric;
import filters.TFIDF;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import util.CommandLineParser;

/**
 * This utility script quickly extracts the good and bad neighbor occurrence
 * frequency arrays for a directory of datasets in the default metric and for
 * the specified neighborhood size. As for optional parameters, there is TFIDF
 * and the option of calculating the secondary shared neighbor distances, namely
 * simcos or simhub. The script doesn't run through the file system recursively,
 * it processes only the datasets on the first level.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DefaultGBHubnessArraysGetter {

    private static int secondaryDistanceOption;
    private static int kSND = 50;
    private static boolean tfidfMode = true;

    /**
     * This script calculates and extracts the good and bad neighbor occurrence
     * frequencies from a directory of files and persists them to a target
     * output directory.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inDir", "Path to the input data directory.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outDir", "Path to the output directory.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-k", "Neighborhood size.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-secondaryDistance", "Whether to use secondary distances."
                + " Possible values: none, simcos, simhub",
                CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        File inDir = new File((String) clp.getParamValues("-inDir").get(0));
        File outDir = new File((String) clp.getParamValues("-outDir").get(0));
        int k = (Integer) clp.getParamValues("-k").get(0);
        String secondaryString =
                (String) clp.getParamValues("-secondaryDistance").get(0);
        secondaryDistanceOption = 0;
        switch (secondaryString.toLowerCase()) {
            case "none":
                secondaryDistanceOption = 0;
                break;
            case "simcos":
                secondaryDistanceOption = 1
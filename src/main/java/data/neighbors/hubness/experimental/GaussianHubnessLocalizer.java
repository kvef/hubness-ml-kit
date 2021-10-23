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
package data.neighbors.hubness.experimental;

import data.generators.MultiDimensionalSphericGaussianGenerator;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.concentration.ConcentrationCalculator;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import feature.correlation.DistanceCorrelation;
import feature.correlation.PearsonCorrelation;
import ioformat.FileUtil;
import ioformat.IOARFF;
import java.io.File;
import java.util.ArrayList;

import java.util.Arrays;
import java.util.Random;
import util.CommandLineParser;

/**
 * This class implements an experiment for tracking hub localization in
 * synthetic Gauss
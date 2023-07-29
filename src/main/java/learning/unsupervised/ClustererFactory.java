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
package learning.unsupervised;

import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.kernel.Kernel;
import distances.kernel.KernelMatrixUserInterface;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import learning.unsupervised.methods.DBScan;
import learning.unsupervised.methods.FastKMeans;
import learning.unsupervised.methods.FastKMeansPlusPlus;
import learning.unsupervised.methods.GHPC;
import learning.unsupervised.methods.GHPKM;
import learning.unsupervised.methods.GKH;
import learning.unsupervised.methods.HarmonicKMeans;
import learning.unsupervised.methods.KMeans;
import learning.unsupervised.methods.KMeansPlusPlus;
import learning.unsupervised.methods.KMedoids;
import learning.unsupervised.methods.KMedoidsPlusPlus;
import learning.unsupervised.methods.KernelGHPKM;
import learning.unsupervised.methods.KernelKMeans;
import learning.unsupervised.methods.LHPC;
import learning.unsupervised.methods.LKH;

/**
 * This class is used to fetch the initial parametrizations of various
 * clustering method objects within the evaluation framework.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClustererFactory {
    
    /**
     * This is how initial clusterer instances are currently generated, as they
   
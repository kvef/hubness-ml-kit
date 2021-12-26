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
package distances.secondary.snd;

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.neighbors.hubness.HubnessExtremesGrabber;
import data.neighbors.hubness.HubnessSkewAndKurtosisExplorer;
import data.neighbors.hubness.HubnessAboveThresholdExplorer;
import data.neighbors.hubness.HubnessVarianceExplorer;
import data.neighbors.hubness.KNeighborEntropyExplorer;
import data.neighbors.hubness.TopHubsClusterUtil;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataSet;
import distances.primary.Com
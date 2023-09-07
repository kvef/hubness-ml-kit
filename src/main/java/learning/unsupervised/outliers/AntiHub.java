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
package learning.unsupervised.outliers;

import data.neighbors.NSFUserInterface;
import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * This class implements the AntiHub outlier detection method proposed in the 
 * paper titled: "Reverse Nearest Neighbors in Unsupervised Distance-Based 
 * Outlier Detection" by Milos Radovanovic et al., that was published in IEEE 
 * Transactions on Knowledge and Data Engineering (TKDE) in 2014.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class AntiHub extends OutlierDetector implements NSFUserInterface {
    
    // The parameter used for summing up the neighbor occurrence frequencies of
    // neighbor points for anti-hub estimation. It is automatically determined
    // within the method, so it does not need to be set manually by the users.
    private float alpha;
    public static final float DEFA
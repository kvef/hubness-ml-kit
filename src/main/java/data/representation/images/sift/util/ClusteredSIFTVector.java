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
package data.representation.images.sift.util;

import data.representation.DataInstance;
import data.representation.images.sift.LFeatVector;
import java.util.Arrays;

/**
 * Same as SIFTVector, with an additional integer attribute that stores the
 * index of the cluster it belongs to.
 *
 * @deprecated
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClusteredSIFTVector extends LFeatVector {

    /**
     * @param rep The embedding data context.
     */
    public ClusteredSIFTVector(ClusteredSIFTRepresentation rep) {
        super(rep);
        setContext(rep);
    }

    /**
     * @param sv The original SIFTVector object that this one is derived from.
     */
    public ClusteredSIFTVector(LFeatVector sv) {
        fAttr = sv.fAttr;
        iAttr = new int[1];
    }

    /**
     * @param instance DataInstance object to take the float values from.
     */
    public ClusteredSIFTVector(DataInstance instance) {
        fAttr = instance.fAttr;
        iAttr = new int[1];
    }

    @Override
    public ClusteredSIFTVector cop
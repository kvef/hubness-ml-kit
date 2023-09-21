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
package linear;

/**
 * This class implements methods for handling linear subspaces that are spanned
 * by a set of vectors.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LinSubspace {

    float[][] defSet = null;
    // The orthonormal basis is kept in a separate structure from the original
    // definition set.
    float[][] basis = null;
    // When used in classification, category is a class label.
    public int category;
    public int currDefSetSize = 0;

    public LinSubspace() {
    }

    /**
     * @param defSet An array of vectors that span the linear subspace.
     */
    public LinSubspace(float[][] defSet) {
        this.defSet = defSet;
    }

    /**
     * @param maxSize Maximum dimensionality of the future linear subspace.
     */
    public LinSubspace(int maxSize) {
        defSet
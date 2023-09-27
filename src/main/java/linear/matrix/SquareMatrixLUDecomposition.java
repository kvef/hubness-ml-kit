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
package linear.matrix;

import combinatorial.Permutation;
import data.representation.util.DataMineConstants;
import java.util.Arrays;

/**
 * LU decomposition of square matrices.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SquareMatrixLUDecomposition {

    private float[][] matrix;
    // Both L and U matrices are encoded here in LUmat, for convenience.
    private float[][] LUmat;
    private int[] perm;
    // Perm is a row permutation 'matrix' - it is encoded in this vector perm 
    // that permutes the rows of matrix during the LU decomposition. This means
    // that PA = LU. Therefore, if invA is sought, invLU must be permuted by P
    // from the right (the columns).
    private boolean decompositionFinished = false;
    private int rank;

    /**
     * The constructor.
     *
     * @param matrix Matrix to decompose.
     */
    public SquareMatrixLUDecomposition(float[][] matrix) {
        this.matrix = matrix;
        if (matrix != null && matrix.length > 0) {
            LUmat = new float[matrix.length][matrix.length];
            // Initializes to a copy of the original.
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix.length; j++) {
                    LUmat[i][j] = matrix[i][j];
                }
            }
            perm = new int[matrix.length];
            for (int i = 0; i < matrix.length; i++) {
                perm[i] = i;
            }
        }
    }

    /**
     * @return Matrix rank after the LU composition has been performed to
     * det
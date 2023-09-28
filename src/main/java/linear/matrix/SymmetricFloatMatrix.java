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

import ioformat.FileUtil;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;

/**
 * In this variant diagonal elements do exist (they are not equal to zero).
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SymmetricFloatMatrix implements DataMatrixInterface {

    private float[][] data = null;

    /**
     * @return Two-dimensional float array that is the row representation of the
     * symmetric float matrix.
     */
    public float[][] getMatrix2DArray() {
        return data;
    }

    public SymmetricFloatMatrix() {
    }

    /**
     * @param dim Number of rows and columns.
     */
    public SymmetricFloatMatrix(int dim) {
        data = new float[dim][];
        for (int i = 0; i < dim; i++) {
            data[i] = new float[dim - i];
        }
    }

    /**
     * @param symmetricMatrix Two-dimensional float array that is the row
     * representation of the symmetric float matrix.
     */
    public SymmetricFloatMatrix(float[][] symmetricMatrix) {
        this.data = symmetricMatrix;
    }

    @Override
    public boolean isSymmetricMatrixImplementation() {
        return true;
    }

    @Override
    public boolean isSquare() {
        return true;
    }

    @Override
    public float getElementAt(int row, int col) {
        if (row <= col) {
            return data[row][col - row];
        } else {
            return data[col][row - col];
        }
    }

    @Override
    public void setElementAt(int row, int col, float newValue) {
        if (row <= col) {
            data[row][col - row] = newValue;
        } else {
            data[col][row - col] = newValue;
        }
    }

    @Override
    public int numberOfRows() {
        return data.length;
    }

    @Override
    public int numberOfColumns() {
        return data.length;
    }

    @Override
    public float[] getRow(int row) {
        // Creates an entire row as if the matrix wasn't only half-filled.
        float[] result = new float[data.length];
        for (int i = 0; i < row; i++) {
            result[i] = data[i][row - i];
        }
        for (int i = row; i < data.length; i++) {
            result[i] = data[row][i - row];
        }
        return result;
    }

    @Override
    public float[] getColumn(int col) {
        // Creates an entire column as if the matrix wasn't only half-filled.
        float[] result = new float[data.length];
        for (int i = 0; i < col; i++) {
            result[i] = data[i][col - i];
        }
        for (int i = col; i < data.length; i++) {
            result[i] = data[col][i - col];
        }
        return result;
    }

    @Override
    public void setRow(int row, float[] newValues) throws Exception {
        for (int i = 0; i < row; i++) {
            data[i][row - i] = newValues[i];
        }
        for (int i = row; i < data.length; i++) {
            data[row][i - row] = newValues[i];
        }
    }

    @Override
    public void setColumn(int col, float[] newValues) throws Exception {
        for (int i = 0; i < col; i++) {
            data[i][col - i] = newValues[i];
        }
        for (int i = col; i < data.length; i++) {
            data[col][i - col] = newValues[i];
        }
    }

    @Override
    public DataMatrixInterface calculateInverse() throws Exception {
        DataMatrixInterface result = new DataMatrix(data.length, data.length);
        if (data.length == 1) {
            if (data[0][0] != 0) {
                result.setElementAt(0, 0, 1 / data[0][0]);
            } else {
                return null;
            }
        }
        int[] indexes;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data.length; j++) {
                indexes = new int[data.length - 1];
                for (int k = 0; k < j; k++) {
                    indexes[k] = k;
                }
                for (int k = j + 1; k < data.length; k++) {
                    indexes[k - 1] = k;
                }
                if (i == 0) {
                    result.setElementAt(i, j, calculateCofactor(indexes, i, 1));
                } else {
                    result.setElementAt(i, j, calculateCofactor(indexes, i, 0));
                }
            }
        }
        // Now calculate determinant from first row minors.
        float detValue = 0;
        for (int i = 0; i < data.length; i++) {
          
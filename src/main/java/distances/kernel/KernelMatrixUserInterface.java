
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
package distances.kernel;

/**
 * This interface is used by learner methods to require kernel matrices.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public interface KernelMatrixUserInterface {

    /**
     * @param kmat float[][] representing the kernel matrix of the training
     * data.
     */
    public void setKernelMatrix(float[][] kmat);

    /**
     * @return float[][] representing the kernel matrix of the training data.
     */
    public float[][] getKernelMatrix();
}
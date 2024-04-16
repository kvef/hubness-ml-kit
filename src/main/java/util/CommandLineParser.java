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
package util;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * This class handles entering command line parameters. Classes that require a
 * large or an arbitrary number of parameters (or parameters where the order is
 * a bit unclear and arbitrary) should use this class for fetching parameters
 * from the command line. The input for a single parameter is assumed to be like
 * this: -param_name::param_val_1,param_val_2,...,param_val_n Different
 * parameter input items are separated by empty spaces.
 *
 * If the class is to be used to also output info() help before main method
 * call, it needs to be given prior information. If not, it can be used to just
 * fetch data easily in arbitrary ordering.
 *
 * If a parameter is encountered twice and multivalued, all encountered values
 * will be used.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CommandLineParser {

    public static final int INIT_SIZE = 50;
    public static final int BOOLEAN = 0;
    public static final int INTEGER = 1;
    public static final int FLOAT = 2;
    public static final int DOUBLE = 3;
    public static final int STRING = 4;
    // To generate error messages if some param doesn't match what's in the hash
    private boolean priorInfo = false;
    private HashMap<String, Integer> paramNamesHash =
            new HashMap(INIT_SIZE, INIT_SIZE);
    private ArrayList<String> paramNamesVect = new ArrayList<>(INIT_SIZE);
    private ArrayList<String> paramDescVect = new ArrayList<>(INIT_SIZE);
    private ArrayList<Integer> paramTypeVect = new ArrayList<>(INIT_SIZE);
    private ArrayList<Boolean> paramObligatoryVect = new ArrayList<>(INIT_SIZE);
    private ArrayLis
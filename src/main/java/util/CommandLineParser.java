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
    private ArrayList<Boolean> paramMultipleValuesVect =
            new ArrayList<>(INIT_SIZE);
    private ArrayList<ArrayList> parsedValues = new ArrayList<>(INIT_SIZE);
    private int paramNum = 0;
    private boolean ignoreExtraParams = false;

    /**
     *
     * @param priorInfo Boolean indicating whether to use prior information for
     * notifying the user of parameter info and for parsing the parameters. If
     * not, the parameters will be fetched automatically, without structure.
     */
    public CommandLineParser(boolean priorInfo) {
        this.priorInfo = priorInfo;
    }

    /**
     * Checks if the specified parameter is already defined in the hash.
     *
     * @param paramName String that is the parameter name.
     * @return True if already defined, false otherwise.
     */
    public boolean hasParamDefinition(String paramName) {
        return paramNamesHash.containsKey(trimLeadingDashes(paramName));
    }

    /**
     * Checks if a value was already parsed for the specified parameter.
     *
     * @param paramName String that is the parameter name.
     * @return True if a value already exists, false otherwise.
     */
    public boolean hasParamValue(String paramName) {
        return hasParamDefinition(trimLeadingDashes(paramName))
                && (parsedValues.get(paramNamesHash.get(trimLeadingDashes(
                paramName))).size() > 0);
    }

    /**
     * This method trims the leading dashes of the parameter name, if provided.
     *
     * @param paramName String that is the parameter name with possible leading
     * dashes.
     * @return String that is the parameter name trimmed of the leading dashes.
     */
    private String trimLeadingDashes(String paramName) {
        int cInd = 0;
        while (cInd < paramName.length() && paramName.charAt(cInd) == '-') {
            cInd++;
        }
        if (cInd == paramName.length()) {
            // The pathological case.
            return "";
        } else if (cInd == 0) {
            return paramName;
        } else {
            return paramName.substring(cInd, paramName.length());
        }
    }

    /**
     * @param ignoreExtraParams Boolean flag indicating whether to ignore when
     * unknown parameters are encountered in the prior info mode or to throw an
     * exception.
     */
    public void setIgnoreExtraParams(boolean ignoreExtraParams) {
        this.ignoreExtraParams = ignoreExtraParams;
    }

    /**
     * Get the parsed values for the specified parameter.
     *
     * @param paramName String that is the parameter name.
     * @return ArrayList of values for the specified parameter.
     * @throws Exception
     */
    public ArrayList getParamValues(String paramName) throws Exception {
        String paramNameTrimmed = trimLeadingDashes(paramName);
        if (!paramNamesHash.containsKey(paramNameTrimmed)) {
            throw new Exception(
                    "Parameter " + paramNameTrimmed
                    + " doesn't exist in the hash");
        }
        int index = (Integer) (paramNamesHash.get(paramNameTrimmed));
        if (parsedValues.get(index).isEmpty()) {
            throw new Exception("Parameter " + paramNameTrimmed
                    + " not provided");
        }
        return parsedValues.get(index);
    }

    /**
     * Parses the command line arguments.
     *
     * @param args String[] representing the command line arguments.
     * @throws Exception
     */
    public void parseLine(String[] args) throws Exception {
        String head, body;
        String[] breakUp;
        for (String s : args) {
            breakUp = s.split("::");
            // Separate parameter name from the values.
            if (breakUp.length < 2) {
                continue;
            }
            head = trimLeadingDashes(breakUp[0]);
            body = breakUp[1];
            // Handle multiple values.
            if (body.contains(",")) {
                breakUp = body.split(",");
            } else {
                breakUp = new String[1];
                breakUp[0] = body;
            }
            if (priorInfo) {
                if (!paramNamesHash.containsKey(head)) {
                    if (ignoreExtraParams) {
                        continue;
                    } else {
                        throw new Exception(
                                "Parameter \"" + head + "\" not defined");
                    }
                }
            } else {
                addParam(head, "", STRING, true, true);
            }
            int paramIndex = paramNamesHash.get(head);
            if (!paramMultipleValuesVect.get(paramIndex)) {
                if (breakUp.length + parsedValues.get(paramIndex).size() > 1) {
                    throw new Exception(
                            "Multiple values for parameter " + head
                            + " not expected");
                }
            }
            // Parse the values into correct types.
            for (int i = 0; i < breakUp.length; i++) {
                switch (paramTypeVect.get(paramIndex)) {
                    case BOOLEAN:
                        parsedValues.get(paramIndex).
                                add(Boolean.parseBoolean(breakUp[i]));
                        break;
                    case INTEGER:
                        parsedValues.get(paramIndex).
                                add(Integer.parseInt(breakUp[i]));
                        break;
                    case FLOAT:
                        parsedValues.get(paramIndex).
                                add(Float.parseFloat(breakUp[i]));
                        break;
                    case DOUBLE:
                        parsedValues.get(paramIndex).
                                add(Double.parseDouble(breakUp[i]));
                        break;
                    case STRING:
                        parsedValues.get(paramIndex).
                                add(breakUp[i]);
                        break;
                }
            }
        }
        // Ensure all the mandatory parameters were provided.
        for (int i = 0; i < paramNum; i++) {
            if (paramObligatoryVect.get(i)) {
                if (parsedValues.get(i).size() < 1) {
                    printInfo();
                    throw new Exception("Obligatory parameter "
                            + paramNamesVect.get(i) + " wasn't provided");
                }
            }
        }
    }

    /**
     * Print out command line parameter specification information.
     */
    public void printInfo() {
        if (priorInfo) {
            System.out.println(paramNum + " parameters");
            for (int i = 0; i < paramNum; i++) {
                System.out.print("-" + paramNamesVect.get(i) + ":: ");
                switch (paramTypeVect.get(i)) {
                    case BOOLEAN:
                        System.out.print("bool -");
                        break;
                    case INTEGER:
                        System.out.print("int -");
                        break;
                    case FLOAT:
                        System.out.print("float -");
                        break;
                    case DOUBLE:
                        System.out.print("double -");
                        break;
                    case STRING:
                        System.out.print("string -");
                        break;
                }
                System.out.prin
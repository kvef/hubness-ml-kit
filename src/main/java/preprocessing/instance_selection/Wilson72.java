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
package preprocessing.instance_selection;

import algref.Author;
import algref.JournalPublication;
import algref.Publication;
import algref.Publisher;
import data.neighbors.NSFUserInterface;
import data.representation.DataSet;
import data.neighbors.NeighborSetFinder;
import java.util.ArrayList;
import distances.primary.CombinedMetric;
import java.util.HashMap;
import java.util.Collections;

/**
 * This class implements an old baseline algorithm described in the paper:
 * Asymptotic Properties of Nearest Neighbor Rules Using Edited Data.
 * Essentially, only those instances that agree with the majority of their
 * nearest neighbors are retained.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Wilson72 extends InstanceSelector implements NSFUserInterface {

    private NeighborSetFinder nsf;
    // Neighborhood size to be used in selection criteria.
    private int kSelection = 1;
    
    @Override
    public Publication getPublicationInfo() {
        JournalPublication pub = new JournalPublication();
        pub.setTitle("Asymptotic Properties of Nearest Neighbor Rules Using "
                + "Edited Data");
        pub.addAuthor(new Author("D. R.", "Wilson"));
        pub.setPublisher(Publisher.IEEE);
        pub.setJournalName("IEEE Transactions on Systems, Man and Cybernetics");
        pub.setYear(1972);
        pub.setStartPage(408);
        pub.setEndPage(421);
        pub.setVolume(2);
        return pu
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
package draw.basic;

import java.awt.*;
import java.awt.image.*;
import java.io.*;
import java.util.Arrays;
import java.util.List;
import javax.imageio.*;
import javax.swing.*;

/**
 * This is a convenience class to create and optionally save to a file a
 * BufferedImage of an area shown on the screen. It covers several different
 * scenarios, so the image can be created of:
 *
 * a) an entire component b) a region of the component c) the entire desktop d)
 * a region of the desktop
 *
 * This class can also be used to create im
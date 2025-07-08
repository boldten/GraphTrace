#!/usr/bin/env python
# coding: utf-8
#
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Martin Boldt, Blekinge Institute of Technology, Sweden.
#
# This file is part of the GraphTrace crime hotspot detection project.
#
# This work has been funded by the Swedish Research Council (grant 2022–05442).
#
# Licensed under the MIT License. You may obtain a copy of the License at:
# https://opensource.org/licenses/MIT
#
# This software is provided "as is", without warranty of any kind, express or
# implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and noninfringement. In no event shall the
# authors or copyright holders be liable for any claim, damages or other
# liability, whether in an action of contract, tort or otherwise, arising
# from, out of or in connection with the software or the use or other dealings
# in the software.
# -----------------------------------------------------------------------------
#

#
# Import of packages
#
import numpy as np
import pandas as pd
import platform
import time
import os
import math
import networkx as nx
import geopandas as gpd
import osmnx
import argparse

import GraphTrace_functions as gt # GraphTrace's helper functions

#
# Main declarations
#
results_path = 'Results/'
city = 'Malmo' # City for slicing data
max_camera_coverage = [50] # Max. allowed crime distance from cluster centers
stage_two_top_n = 1000 # Only consider top N positions with most crimes (set to -1 to consider all, might be very computational expensive)
stage_two_spacing = 5 # meters between grid cells in 2nd stage analysis
min_cluster_size = 20 # only consider positions with at least this many crimes
show_top_n_results = 10 # Show N best results in runtime output
gt.verbose = 1 # 0: Only basic output, 1: additional progress info, 2: debugging

#
# Start of main execution
#
if __name__ == "__main__":
    gt.show_disclaimer()
    col_names = ['Method', 'Years', 'Distance', 'Execution_time']

    parser = argparse.ArgumentParser(description="Run GraphTrace with a specified CSV file.")
    parser.add_argument("csv_file", help="Path to the input crime data CSV file")
    args = parser.parse_args()
    crime_data_path = args.csv_file

    years_training = gt.check_input_file( crime_data_path )

    # Check if directory exists, otherwise create them
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for year in years_training:
        evaluation_year = year + 1  #evaluate against next upcoming year
        years_string = f"{year}->{evaluation_year}"
        print( "===================================================[ {:^12} ]==================================================".format( city ) )
        print( "===================================================[ {:^12} ]==================================================".format( f"{year}->{evaluation_year}" ) )
        unique_position_crime_counts, all_crimes_gdf = gt.create_geodataframe( crime_data_path, year, False )
        unique_position_eval_crimes, eval_crimes = gt.create_geodataframe( crime_data_path, evaluation_year, False )

        gt.print_verbose( 0, "x Number of crimes  = {:>6}".format( all_crimes_gdf.shape[0] ) )
        gt.print_verbose( 0, "x Unique positions  = {:>6}".format( unique_position_crime_counts.shape[0] ) )
        gt.print_verbose( 0, "x Crimes / Uniq.pos = {:>6.3f}".format( all_crimes_gdf.shape[0] / unique_position_crime_counts.shape[0] ) )        

        pai_A = gt.calculate_area_gdf( eval_crimes ) # Calculate the full study area using a rectangle that includes all crimes over all years
        gt.print_verbose( 0, "x Study area (km^2) = {:>6}".format( round(pai_A,0) ) )
        
        for current_max_camera_coverage in max_camera_coverage:
            print( "======================================================[ {:^4}m ]======================================================".format( current_max_camera_coverage ) )
            pai_a = gt.calculate_camera_area( current_max_camera_coverage )
            gt.print_verbose( 0, "x CCTV coverage (km^2) = {:>5}".format( round(pai_a,4) ) )
        
            unique_position_crime_counts, all_crimes_gdf = gt.create_geodataframe( crime_data_path, year, False )
            unique_position_eval_crimes, eval_crimes = gt.create_geodataframe( crime_data_path, evaluation_year, False )
            print( "--------------------------------------[ Location detection method = GraphTrace ]--------------------------------------" ) 
            identified_positions_df = None
            total_start = time.time()
            
            # path where the resulting graphs is stored
            graph_file = f"{results_path}/GraphTrace_graph_for_city={city}_method=GraphTrace_year={year}_dist={current_max_camera_coverage}m.pickle" 
            G, identified_positions_df = gt.run_graph_method( unique_position_crime_counts.copy(), year, current_max_camera_coverage, all_crimes_gdf.copy(), graph_file, stage_two_top_n, stage_two_spacing, "extended", min_cluster_size )
            identified_positions_df = gt.verify_position_crime_counts_generic(identified_positions_df, all_crimes_gdf, current_max_camera_coverage)
            identified_positions_df = gt.calculate_pai_index(identified_positions_df, eval_crimes, pai_a, pai_A)
            
            total_elapsed = time.time()-total_start
            
            identified_positions_df = gt.evaluate_positions_against_next_year(identified_positions_df, eval_crimes, current_max_camera_coverage, evaluation_year, pai_a, pai_A)
            
            gt.print_results( identified_positions_df, total_elapsed, current_max_camera_coverage, evaluation_year, show_top_n_results )

            
            result_file = results_path + f"/Results_city={city}_train-year={year}_eval-year={evaluation_year}_method=GraphTrace_distance={current_max_camera_coverage}m.csv"
            gt.store_csv_file( result_file, identified_positions_df )
            gt.print_verbose( 1, f"* Stored results to CSV file: {result_file}" )
                        
            print( "" )

    print( "Finished." )
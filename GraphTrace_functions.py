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

import numpy as np
import pandas as pd
import math
import time
import sys
import os
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from haversine import haversine
from scipy.spatial.distance import cdist
from sklearn.neighbors import BallTree
import geopy.distance
import matplotlib.pyplot as plt
import pickle

EARTH_RADIUS = 6371000.0 # Used for conversion between degrees and meters
STR_WIDTH = 110 # Approximate string output width
INPUT_COLUMNS = {'crime_code', 'year', 'latitude', 'longitude'} # Required columns in input CSV file

def show_disclaimer():
    print( """
    DISCLAIMER:
    This software is provided "as is", without warranty of any kind, express or implied, including but not limited to
    the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the
    author(s) or affiliated institution(s) be liable for any claim, damages or other liability, whether in an action
    of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings
    in the software.

    This code is intended for research and educational purposes only. It is not intended for operational use, nor
    should it be used for decision-making in law enforcement or public safety contexts without independent validation
    and appropriate oversight.

    Use at your own risk.
    """+"\n" )

def check_input_file( crime_data_path ):
    """
    Check so that the CSV file can be opened, has the required columns, and includes at least two
    years for crime data. Use all but the last year's data for detecting hotspots, and evaluate 
    against the next year's data. E.g., if 3 years of data is include, hotspots are detected using
    1st and 2nd year's data independently, and these are evaluated against 2nd and 3rd respectively.
    """
    if not os.path.isfile(crime_data_path):
        print(f"\n[Error] File not found: {crime_data_path}")
        print("Please check the filename and try again.\n")
        sys.exit(1)
    try:
        crimes_df = pd.read_csv( crime_data_path )
        missing_columns = INPUT_COLUMNS - set(crimes_df.columns)
        if missing_columns:
            print(f"Error: Missing required column(s): {', '.join(missing_columns)}")
            sys.exit(1)
    except Exception as e:
        print(f"Error while reading file: {crime_data_path}. Error: {e}")
        print("Exiting.")
        sys.exit(1)

    # Ensure there the crime data spans at least two years
    unique_years = sorted(crimes_df['year'].dropna().unique())
    if len(unique_years) < 2:
        print("Error: The dataset must contain at least two unique years.")
        sys.exit(1)

    # Select all but the last year for training
    years_training = unique_years[:-1]

    return years_training

def print_verbose( asked_verbose_level, string, ending="\n", flushing=False ):
    if ( verbose >= asked_verbose_level ):
        print( string, end=ending, flush=flushing )
        return len(string)

def store_csv_file(file, df):
    export_columns = {
        'rank': 'Rank',
        'latlong_geometry': 'Coordinates',
        'eval_total_count': 'NextYearCrimeCount',
        'eval_pai': 'NextYearPAI',
        'total_count': 'SameYearCrimeCount',
        'pai': 'SameYearPAI'
    }
    df[export_columns.keys()].rename(columns=export_columns).to_csv(file, index=False)
    
def read_from_pickle(file):
    try:
        pikd = open(file, 'rb')
        data = pickle.load(pikd)
        pikd.close()
    except:
        data = None
        pass
    return data

def distance_matrix(points):
    """
    Calculate distance matrix based on Haversine distance between shapely Point objects.
    """
    n = len(points)
    coords = np.zeros((n, 2))
    for i, p in enumerate(points):
        coords[i, 0] = p.y
        coords[i, 1] = p.x
    return cdist( coords, coords, haversine, unit='m' )


def preprocess_data( df, show_output = True ):
    if show_output:
        print_verbose( 1, "* Preprocess data ... ", "" )

    df.index.name = 'id'        # Set the index name
    df.reset_index(inplace=True) # set up as index

    df = df.dropna(axis=0,how='any',subset=['latitude','longitude'])
    df['latitude'] = pd.to_numeric(df['latitude'])
    df['longitude'] = pd.to_numeric(df['longitude'])
    df = gpd.GeoDataFrame( df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326 )
    df = df[['crime_code', 'year', 'latitude', 'longitude', 'geometry']]

    if show_output:
        print_verbose( 1, " done." )
        
    return df


def create_node_count_attributes( G, unique_position_crime_counts ):
    print_verbose( 1, "* Creating node's count attribute ... ", "" )
    no_hits = 0
    for node in G.nodes:
        pt = Point(G.nodes[node]['lon'], G.nodes[node]['lat']) # Create the Point object (Point(x=lon, y=lat))
        if pt in unique_position_crime_counts.index:
            current_node_crime_count = int( unique_position_crime_counts.loc[pt].iloc[0] )
            nx.set_node_attributes( G, {node: current_node_crime_count}, name='count')
        else:
            no_hits += 1

    print_verbose( 1, " done. " )
    return G

def create_edge_weights_and_node_counts(G, unique_position_crime_counts):
    print_verbose( 1, "* Creating edge weights and node counts ... ", "" )
    for edge in G.edges():
        node1, node2 = edge
        try:
            weight = G.nodes[node1]['count'] + G.nodes[node2]['count']
            G[edge[0]][edge[1]]['weight'] = weight
        except Exception as e:            
            print(e)
            sys.exit(1)
        
    tmp_positions_data = []
    
    for node in G.nodes():
        total_count = G.degree(node, weight='weight') - (G.degree(node)- 1) * G.nodes[node]['count']
        nx.set_node_attributes( G, {node: total_count}, name='total_count')

        neighbors_tuple_list = []
        for curr_neighbor in list(G.neighbors(node)):
            neighbors_tuple_list.append( ( curr_neighbor, unique_position_crime_counts.iloc[curr_neighbor].name ) ) # add (node id, Shapely Point)
            
        current_pos_dict = {
            'node': node,
            'total_count': G.nodes[node]['total_count'],
            'total_count_with_vsa': -1,
            'degree': G.degree(node),
            'count': G.nodes[node]['count'],                
            'neighbors': neighbors_tuple_list,
            'neighbors_removed': list(),
            'geometry': None
           }

        tmp_positions_data.append( current_pos_dict )
    
    identified_positions = pd.DataFrame( tmp_positions_data ).sort_values(by=['total_count', 'count'], ascending=False )

    print_verbose( 1, " done. " )
    
    return ( G, identified_positions )

def prune_positions_by_minimum_threshold( identified_positions_df, min_cluster_size, max_num_positions_to_consider ):
    print_verbose( 1, f"* Pruning by only keeping positions with a total count >= {min_cluster_size} (but max {max_num_positions_to_consider} positions) ... ", "" )    
  
    identified_positions_df = identified_positions_df[identified_positions_df[ 'total_count' ] >= min_cluster_size ]
    identified_positions_df = identified_positions_df.sort_values(by=['total_count'], ascending=False )
    identified_positions_df.reset_index(drop=True, inplace=True) # Make sure index=0 is the best postion, 1 is the 2nd best etc.    

    if identified_positions_df.shape[0] > max_num_positions_to_consider:
        identified_positions_df = identified_positions_df.head( max_num_positions_to_consider )
        
    print_verbose( 1, " done. " )
    
    return identified_positions_df




def verify_position_crime_counts_generic( positions_df, all_crimes_gdf, current_max_camera_coverage ):
    print_verbose( 1, f"* Verify number of crimes per location (N={positions_df.shape[0]}) within same year ... ", "" )
    
    all_crimes = all_crimes_gdf.copy()
    
    all_crimes['latlong_geometry'] = all_crimes['geometry']
    positions_df['latlong_geometry'] = positions_df['geometry']
        
    if all_crimes.crs is None or all_crimes.crs != "EPSG:32633":  # EPSG:4326==WGS84, EPSG:3006==SWEREF99/TM
        all_crimes.crs = "EPSG:4326"
        all_crimes.to_crs(epsg=32633, inplace=True)

    if positions_df.crs is None or positions_df.crs != "EPSG:32633":
        positions_df.crs = "EPSG:4326"
        positions_df.to_crs(epsg=32633, inplace=True)
    
    positions_df['members'] = None
    positions_df['verified_count'] = -1
    already_counted_crime_points = set()

    # Calculate the number of events within radius_m meters for each position
    for index, position in positions_df.iterrows():
        buffer = position.geometry.buffer( current_max_camera_coverage+0.5 ) # add small buffer to account for rounding errors
        crime_points = all_crimes[all_crimes.within(buffer)]['latlong_geometry'].unique()
        
        # Exclude crime points already counted
        unique_crime_points = [p for p in crime_points if p not in already_counted_crime_points]
        
        verified_count = 0
        for point in unique_crime_points:
            try:
                current = unique_position_crime_counts.loc[ point ]['count']
            except:
                current = 0
            verified_count += current
        
        positions_df.at[ index, 'members' ] = list( unique_crime_points )        
        positions_df.at[ index, 'verified_count' ] = verified_count
        already_counted_crime_points.update( unique_crime_points )

    positions_df = positions_df.sort_values(by=['verified_count'], ascending=False )
    positions_df[ 'verified_count' ] = positions_df[ 'verified_count' ].astype( int )

    print_verbose( 1, f" done." )
    
    return positions_df


def create_geodataframe( csv_path, year, show_output = True ):
    try:
        crimes_df = pd.read_csv( csv_path )
        crimes_df = crimes_df[ crimes_df.year == year] 
        crimes_df = preprocess_data( crimes_df, show_output )
    except Exception as e:
        print(f"An error occurred when reading the crime data from: {csv_path}.")
        print("The error was:", e)
    
    unique_position_crime_counts = pd.DataFrame( crimes_df['geometry'].value_counts(sort=False) )
    unique_position_crime_counts.columns = ['count']
    
    return unique_position_crime_counts, crimes_df
    
def grid_haversine_distance(coord1, coord2):
    """
    Haversine formula to calculate the great circle distance in meters between two points.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    distance = 6371000 * c
    return distance

def calculate_area_gdf(gdf):
    lats = gdf.geometry.y
    lons = gdf.geometry.x
    top_right = (max(lats), max(lons))
    bottom_left = (min(lats), min(lons))
    width = grid_haversine_distance((bottom_left[0], top_right[1]), top_right)
    height = grid_haversine_distance(bottom_left, (bottom_left[0], top_right[1]))
    A = width * height
    A = A/1000000   # Return area in km^2
    
    return A

def calculate_camera_area( radius_m ):
    a = math.pi * radius_m**2
    a = a/1000000   # Return area in km^2
    return a

def calculate_pai_index( df, eval_crimes, pai_a, pai_A ):
    #pai = n/N / a/A
    df['pai'] = (df['total_count'] / eval_crimes.shape[0]) / (pai_a / pai_A)
    
    return df
    
def evaluate_positions_against_next_year(positions_gdf, eval_crimes, radius_meters, eval_year, pai_a, pai_A):
    if positions_gdf.empty:
        print_verbose(1, f"Warning: No hotspot positions provided for year {eval_year}. Skipping evaluation.")
        positions_gdf['eval_total_count'] = []
        positions_gdf['eval_pai'] = []
        positions_gdf['rank'] = []
        return positions_gdf

    # Ensure positions and crimes are in EPSG:4326 (lat/lon)
    if positions_gdf.crs != "EPSG:4326":
        positions_gdf = positions_gdf.set_geometry("latlong_geometry")
        positions_gdf.set_crs("EPSG:4326", inplace=True, allow_override=True)

    if eval_crimes.crs != "EPSG:4326":
        eval_crimes = eval_crimes.to_crs(epsg=4326)

    # Prepare position coordinates (lat, lon → radians)
    position_coords = np.radians([(pt.y, pt.x) for pt in positions_gdf.geometry])

    # Prepare eval crime coordinates (lat, lon → radians)
    eval_coords = np.radians([(pt.y, pt.x) for pt in eval_crimes.geometry])

    # Build BallTree from evaluation year crimes
    tree = BallTree(eval_coords, metric='haversine')
    radius_rad = radius_meters / EARTH_RADIUS

    # Query number of crimes within radius for each position
    eval_counts = tree.query_radius(position_coords, r=radius_rad, count_only=True)

    # Store results
    positions_gdf['eval_total_count'] = eval_counts
    positions_gdf['eval_pai'] = (positions_gdf['eval_total_count'] / eval_crimes.shape[0]) / (pai_a / pai_A)
    
    positions_gdf = positions_gdf.sort_values(by='eval_total_count', ascending=False).reset_index(drop=True)
    positions_gdf['rank'] = positions_gdf.index + 1

    return positions_gdf

def run_graph_method(unique_position_crime_counts, year, current_max_camera_coverage, all_crimes_gdf, graph_file, stage_two_top_n=1000, stage_two_spacing=5, search_type="extended", min_cluster_size=20):
    """
    Run Graph Method: Executes the graph method to build a NetworkX graph based
    on unique crime positions (unique lat/lon pairs in the dataset). If 'search_type' 
    is set to "basic" these positions are returned as a DataFrame sorted on column
    'total_count' (in descending order). If 'search_type' is set to "extended" then a 
    predefined pattern of points around the suggested hotspots are also analyzed and 
    added to the resulting DataFrame as potential hotspots.

    Parameters:
    - unique_position_crime_counts: DataFrame with 'geometry' index set to 
    Point(lat,Lon) and column 'count' being the number of crimes in that position.
    - year: current year analysed in the dataset.
    - current_max_camera_coverage: maximum radius (in meters) to consider crimes within.
    - all_crimes_gdf: GeoDataFrame with all crime data and geometries.
    - graph_file: Path to the graph file.
    - stage_two_top_n: number of top positions from stage one to perform extended search 
    over, the larger the better results but also slower execution (default 1000).
    - stage_two_spacing: distance (in meters) between grid points during stage two search,
    default is 5 meters.
    - search_type: "extended" (default) includes stage-two extended search of positions 
    around the suggested hotspots from stage-one (graph) approach, "basic" only 
    executes the graph approach without extended stage two search".
    - min_cluster_size: Min number of crimes required to form a cluser, default 20.

    Returns:
    - DataFrame of selected hotspots with covered crime counts. The columns are as follows:
        o Rank: Rank of position starting at 1.
        o Coordinates: Point position for the hotspot, e.g., POINT (13.0154 55.6171)
        o NextYearCrimeCount: Crime count of hotspot calculated on next year's data, i.e., an integer such as 40
        o NextYearPAI: PAI index for the hotspot calculated on next year's data, i.e., float value such as 50.3185
        o SameYearCrimeCount: The hotspot's crime count calculated on the same year's data.
        o SameYearPAI: The hotspot's PAI calculated on the same year's data.
    """

    start = time.time()
    
    print_verbose(0, f"* Creating graph:")
    
    geo_point_lookup = unique_position_crime_counts.index.unique()
    G = create_graph_with_balltree(geo_point_lookup, current_max_camera_coverage)

    # After G is ready, continue with your existing logic
    G = create_node_count_attributes(G, unique_position_crime_counts)
    G, identified_positions_df = create_edge_weights_and_node_counts(G, unique_position_crime_counts)
    
    for index, row in identified_positions_df.iterrows():       
        unique_pos_idx = identified_positions_df.iloc[ index ]['node']
        point = unique_position_crime_counts.iloc[unique_pos_idx]
        identified_positions_df.at[ index, 'latitude' ] = point.name.y
        identified_positions_df.at[ index, 'longitude' ] = point.name.x
        identified_positions_df.at[ index, 'geometry' ] = point.name
    
    identified_positions_df = identified_positions_df.reset_index( drop=True )
    
    
    identified_positions_df = gpd.GeoDataFrame( identified_positions_df, geometry=gpd.points_from_xy( identified_positions_df.longitude, identified_positions_df.latitude ) )
    
    num_positions_before_pruning = identified_positions_df.shape[0]
    identified_positions_df = prune_positions_by_minimum_threshold( identified_positions_df, min_cluster_size, stage_two_top_n
     )
    print_verbose( 0, f"   - Pruned away {num_positions_before_pruning-identified_positions_df.shape[0]} positions." )
    
    
        
    if search_type == "extended":
        print_verbose( 1, "* Stage two: extended search around hotspots found in stage one." )
        
        print_verbose( 1, "  - Generating lat/lon offset mask for stage two extended search around hotspots found in stage one: ...", "" )
        stage2_offset_mask = generate_hexagonal_offset_mask(current_max_camera_coverage, stage_two_spacing, center_lat=55.6, center_lon=13.0)

        print_verbose( 1, f"  - Starting stage two extended search (N={stage_two_top_n}): ", flushing=True )
        sys.stdout.flush()
        stage_two_hotspots_df = stage_two_extended_search( identified_positions_df, stage2_offset_mask, all_crimes_gdf, current_max_camera_coverage, year, stage_two_top_n )
        print_verbose( 1, f"  - Improved best hotspot crime count from {identified_positions_df.head(1).total_count.iloc[0]} -> {stage_two_hotspots_df.head(1).total_count.iloc[0]} (a total of {stage_two_hotspots_df.shape[0]} hotspots found).")                
        
        print_verbose( 1, "* Ranking hotspots by maximized crime coverage (one crime can only be accounted to one hotspot): ", "", flushing=True )
        final_hotspots_df = rank_hotspots(stage_two_hotspots_df, all_crimes_gdf, current_max_camera_coverage, min_cluster_size)
        
        final_hotspots_df = final_hotspots_df.sort_values(by='total_count', ascending=False).reset_index(drop=True)
        final_hotspots_df = gpd.GeoDataFrame(final_hotspots_df, geometry='geometry', crs='EPSG:4326')

        return G, final_hotspots_df

    elif search_type == "basic":
        print_verbose( 1, "* Ranking hotspots by maximized crime coverage (each crime can only be accounted to one hotspot): ", "", flushing=True )
        identified_positions_df = rank_hotspots(identified_positions_df, all_crimes_gdf, current_max_camera_coverage, min_cluster_size, allow_crime_recorded_in_multiple_hotspots=allow_duplicate_counts)        
        final_hotspots_df = rank_hotspots(stage_two_hotspots_df, all_crimes_gdf, current_max_camera_coverage, min_cluster_size)
        final_hotspots_df = gpd.GeoDataFrame(final_hotspots_df, geometry='geometry', crs='EPSG:4326')

        return G, identified_positions_df
    else:
        print_verbose( 1, f"ERROR: run_graph_method() got unknown 'search_type' set to \"{search_type}\"  " )
        sys.exit(1)
        
def create_graph_with_balltree(unique_positions, radius_meters):
    """
    Create a graph using BallTree to efficiently find nearby crimes.
    
    Parameters:
    - unique_positions: pandas Index of shapely Points (crimes).
    - radius_meters: maximum connection distance (meters).

    Returns:
    - NetworkX Graph with crimes as nodes, and distances as edge weights.
    """
    print_verbose(1, "  - Building graph with BallTree ... ", "")

    # Convert to (lat, lon) -> radians
    coords_deg = np.array([(p.y, p.x) for p in unique_positions])
    coords_rad = np.radians(coords_deg)

    tree = BallTree(coords_rad, metric='haversine')
    radius_radians = radius_meters / EARTH_RADIUS  # Earth radius in meters

    # Query all neighbors within radius
    indices_array, distances_array = tree.query_radius(coords_rad, r=radius_radians, return_distance=True, sort_results=True)

    # Create the graph
    G = nx.Graph()
    
    # Add nodes
    for i, point in enumerate(unique_positions):
        G.add_node(i, lat=point.y, lon=point.x)

    # Add edges with distances
    for i, (neighbors, distances) in enumerate(zip(indices_array, distances_array)):
        for j, dist_rad in zip(neighbors, distances):
            if i < j:  # avoid duplicate edges (because BallTree gives both directions)
                dist_meters = dist_rad * 6371000  # convert back to meters
                G.add_edge(i, j, weight=dist_meters)

    print_verbose(1, "done.")
    return G

def generate_hexagonal_offset_mask(radius_meters, spacing_meters, center_lat, center_lon):
    crs_wgs84 = "EPSG:4326"
    crs_projected = "EPSG:3857"

    # Project center point to metric CRS
    center = gpd.GeoDataFrame(geometry=[Point(center_lon, center_lat)], crs=crs_wgs84)
    center_proj = center.to_crs(crs_projected)
    center_x, center_y = center_proj.geometry[0].x, center_proj.geometry[0].y

    # Estimate number of rings and adjust radial spacing to match outer radius
    ideal_step = spacing_meters * np.sqrt(3) / 2
    num_rings = max(1, round(radius_meters / ideal_step))
    adjusted_radial_step = radius_meters / num_rings

    offsets = []
    for ring in range(num_rings + 1):
        r = ring * adjusted_radial_step
        if r == 0:
            offsets.append(Point(center_x, center_y))
            continue

        circumference = 2 * np.pi * r
        num_points = max(6, int(np.ceil(circumference / spacing_meters)))
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            offsets.append(Point(x, y))

    # Convert back to lat/lon and compute deltas
    offset_gdf = gpd.GeoDataFrame(geometry=offsets, crs=crs_projected).to_crs(crs_wgs84)
    delta_offsets = []
    for pt in offset_gdf.geometry:
        delta_lat = round(pt.y - center_lat, 5)
        delta_lon = round(pt.x - center_lon, 5)
        delta_offsets.append((delta_lat, delta_lon))

    unique_offsets = np.unique(delta_offsets, axis=0)
    

    print_verbose(1, f" done (mask contains {len(unique_offsets)} points).")
    return unique_offsets


def stage_two_extended_search(hotspots_df, offset_mask, all_crimes_df, radius_meters, year, top_n):
    """
    Efficiently filters unique hotspots using BallTree for spatial lookup.
    Vectorized implementation for better performance.
    """
    best_hotspots = []
    if top_n > 0:
        stage1_hotspots_df = hotspots_df.sort_values(by='total_count', ascending=False).head(top_n)
    else:
        stage1_hotspots_df = hotspots_df.sort_values(by='total_count', ascending=False)

    # Convert crime locations to radians (lat, lon in radians)
    crime_coords_rad = np.radians([
        (round(p.y, 5), round(p.x, 5)) for p in all_crimes_df['geometry']
    ])
    tree = BallTree(crime_coords_rad, metric='haversine')
    radius_radians = radius_meters / EARTH_RADIUS  # Earth radius in meters

    for _, row in stage1_hotspots_df.iterrows():
        center = row['geometry']
        center_lat, center_lon = round(center.y, 5), round(center.x, 5)

        # Generate offset positions (lat, lon), with rounding
        offset_positions = [
            (round(center_lat + offset[0], 5), round(center_lon + offset[1], 5))
            for offset in offset_mask
        ]

        # Convert to radians and swap to (lon, lat) for BallTree
        query_coords_rad = np.radians(offset_positions)
        

        # Batch query all offsets
        counts = tree.query_radius(query_coords_rad, r=radius_radians, count_only=True)

        current_best_positions = []
        higher_found = False

        for pos, count in zip(offset_positions, counts):
            if count > row['total_count']:
                higher_found = True
                current_best_positions.append({
                    'total_count': count,
                    'geometry': Point(round(pos[1], 5), round(pos[0], 5))  # Store as (lon, lat)
                })

        if not higher_found:
            current_best_positions.append({
                'total_count': row['total_count'],
                'geometry': Point(center_lon, center_lat)
            })

        best_hotspots.extend(current_best_positions)
        print_verbose(1, ".", "", flushing=True)

    print_verbose(1, f" done.","")
    print_verbose(1, "")

    # Return deduplicated and sorted result
    best_hotspots_df = pd.DataFrame(best_hotspots).sort_values(by='total_count', ascending=False).drop_duplicates(subset='geometry', keep='first').reset_index(drop=True)
    
    return best_hotspots_df

def rank_hotspots(candidate_hotspots_df, all_crimes_gdf, radius_meters, min_cluster_size):
    """
    Rank Hotspots: Selects a subset of positions that do not share crimes and
    maximize crime coverage, where each selected hotspot covers at least
    `min_cluster_size` crimes and each crime is counted only once.

    Parameters:
    - candidate_hotspots_df: DataFrame with 'geometry' column as (lon, lat) tuples.
    - all_crimes_gdf: GeoDataFrame with crime data and geometries.
    - radius_meters: Radius within which crimes are considered "covered".
    - min_cluster_size: Minimum number of crimes required to keep a hotspot.
    - method: "extended" (default) includes state-two extended search of positions 
    around suggested hotspots from stage-one (graph) approach, "stage-one" only 
    executes the graph approach without extended stage two search".

    Returns:
    - DataFrame of selected hotspots with covered crime counts.
    """
    progress_output = 20 #Write progress N times, i.e., 20 means output every 1/20 (5%) of the execution
    
    crime_coords_deg = np.array([
        (round(p.y, 5), round(p.x, 5)) for p in all_crimes_gdf['geometry']
    ]) 
    crime_coords_rad = np.radians(crime_coords_deg)
    tree = BallTree(crime_coords_rad, metric='haversine')
    radius_radians = radius_meters / EARTH_RADIUS

    total_candidates = len(candidate_hotspots_df)
    progress_interval = max(total_candidates // progress_output, 1) 
    
    sorted_hotspot_indices = candidate_hotspots_df['total_count'].sort_values(ascending=False).index.tolist()
    
    hotspot_coverage = {}
    for idx, row in candidate_hotspots_df.iterrows():
        lat, lon = row['geometry'].y, row['geometry'].x
        pt_rad = np.radians([[lat, lon]])  # BallTree wants (lat, lon)

        indices = tree.query_radius(pt_rad, r=radius_radians)[0]
        hotspot_coverage[idx] = set(indices)
        
        if (idx + 1) % progress_interval == 0:
            print_verbose( 1,".", "")
    
    assigned_crimes = set()
    selected_hotspots = []
    while True:
        best_idx = None
        best_gain = 0
        best_coverage = set()

        for hs_idx in sorted_hotspot_indices:
            if hs_idx not in hotspot_coverage:
                continue

            covered = hotspot_coverage[hs_idx]
            
            new_crimes = covered - assigned_crimes
            
            if len(new_crimes) >= min_cluster_size:
                best_idx = hs_idx
                best_gain = len(new_crimes)
                best_coverage = new_crimes
                break  # Pick first valid candidate

        if best_idx is None:
            break  # No more good hotspots left

        assigned_crimes.update(best_coverage)
        
        selected_hotspots.append({
            'total_count': best_gain,
            'geometry': candidate_hotspots_df.loc[best_idx, 'geometry']
        })
        del hotspot_coverage[best_idx]

        if len(selected_hotspots) % 100//progress_output == 0:
            print_verbose(1, ".", "", flushing=True)
        
    total_crimes = len(crime_coords_deg)
    assigned_count = len(assigned_crimes)
    unassigned_count = total_crimes - assigned_count

    final_hotspot_result_df = pd.DataFrame(selected_hotspots)
    print_verbose( 1, f" done." )
    print_verbose( 1, f"  - A total of {final_hotspot_result_df.shape[0]} hotspots found." )
    print_verbose( 1, f"  - Assigned {assigned_count} crimes to {len(selected_hotspots)} hotspots.")
    print_verbose( 1, f"  - {unassigned_count} crimes remain unassigned.")

    return final_hotspot_result_df

def print_results( verified_gdf, time_duration, radius, evaluation_year, show_top_n_results ):
    t_str = "[ {}.{} ]".format( time.strftime("%Hh%Mm%Ss",time.gmtime( time_duration ) ), str(time_duration%1000).split('.')[1][0:3] )
    if verified_gdf is None:
        print_verbose( 0, "* Number of crimes within range in top-0 positions (out of 0):" )
        print_verbose( 0, "   - Empty list." )        
        print_verbose( 0, "* Total time:  {0: >{1}}".format( t_str, STR_WIDTH-8  ) )
    else:
        print_verbose( 0, f"* Crime counts for the top-{show_top_n_results} positions (out of {verified_gdf.shape[0]})." )
        print_verbose( 0, f"   - For (i) all crimes within {radius}m in the next year's data ({evaluation_year}), and (ii) PAI index for those crimes." )
        
        verified_gdf = verified_gdf.sort_values(by='eval_total_count', ascending=False)
        print_verbose( 0, "      (i)   {}".format( list( verified_gdf.head( show_top_n_results ).eval_total_count ) ) )        
        print_verbose( 0, "      (ii)  {}".format( list( map(int, round(verified_gdf.head( show_top_n_results ).eval_pai,0) ) ) ) )
        print_verbose( 0, f"     Mean crime count on evaluation data = {verified_gdf.loc[:, 'eval_total_count'].mean():.3f} (SD: {verified_gdf.loc[:, 'eval_total_count'].std():.3f})" )
        print_verbose( 0, f"     Mean PAI index on evaluation data = {verified_gdf.loc[:, 'eval_pai'].mean():.3f} (SD: {verified_gdf.loc[:, 'eval_pai'].std():.3f})" )
        
        print_verbose( 0, f"   - For (iii) all crimes within {radius}m in the training data, and (iv) PAI index for those crimes." )
        verified_gdf = verified_gdf.sort_values( by='verified_count', ascending=False )
        print_verbose( 0, "      (iii) {}".format( list( verified_gdf.head( show_top_n_results ).total_count ) ) )        
        print_verbose( 0, "      (iv)  {}".format( list( map(int, round(verified_gdf.head( show_top_n_results ).pai,0) ) ) ) )
        print_verbose( 0, f"     Mean crime count on the training data = {verified_gdf.loc[:, 'total_count'].mean():.3f} (SD: {verified_gdf.loc[:, 'total_count'].std():.3f})" )
        print_verbose( 0, f"     Mean PAI index on the training data =   {verified_gdf.loc[:, 'pai'].mean():.3f} (SD: {verified_gdf.loc[:, 'pai'].std():.3f})" )
                
        print_verbose( 0, "* Total time:  {0: >{1}}".format( t_str, STR_WIDTH-8  ) )

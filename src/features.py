"""
Feature extraction utilities for bird radar track classification.
Extracts features from trajectory, temporal, and radar signature data.
"""

import pandas as pd
import numpy as np
from shapely import wkb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def parse_trajectory(wkb_hex):
    """Parse WKB-encoded trajectory and extract spatial features."""
    try:
        if pd.isna(wkb_hex):
            return None
        geom = wkb.loads(wkb_hex, hex=True)
        coords = list(geom.coords)
        return coords
    except:
        return None


def extract_trajectory_features(coords):
    """Extract statistical features from trajectory coordinates."""
    if coords is None or len(coords) == 0:
        return {}
    
    coords_array = np.array(coords)
    
    # Basic statistics
    features = {
        'traj_length': len(coords),
        'traj_x_mean': np.mean(coords_array[:, 0]),
        'traj_x_std': np.std(coords_array[:, 0]),
        'traj_x_min': np.min(coords_array[:, 0]),
        'traj_x_max': np.max(coords_array[:, 0]),
        'traj_y_mean': np.mean(coords_array[:, 1]),
        'traj_y_std': np.std(coords_array[:, 1]),
        'traj_y_min': np.min(coords_array[:, 1]),
        'traj_y_max': np.max(coords_array[:, 1]),
        'traj_z_mean': np.mean(coords_array[:, 2]) if coords_array.shape[1] > 2 else 0,
        'traj_z_std': np.std(coords_array[:, 2]) if coords_array.shape[1] > 2 else 0,
    }
    
    # Distance features
    if len(coords) > 1:
        distances = np.sqrt(np.sum(np.diff(coords_array[:, :2], axis=0)**2, axis=1))
        features['traj_total_distance'] = np.sum(distances)
        features['traj_avg_step_distance'] = np.mean(distances)
        features['traj_max_step_distance'] = np.max(distances)
        features['traj_distance_std'] = np.std(distances)
        
        # Straightness (ratio of straight-line distance to total path length)
        straight_distance = np.sqrt((coords_array[-1, 0] - coords_array[0, 0])**2 + 
                                   (coords_array[-1, 1] - coords_array[0, 1])**2)
        features['traj_straightness'] = straight_distance / (features['traj_total_distance'] + 1e-6)
    else:
        features['traj_total_distance'] = 0
        features['traj_avg_step_distance'] = 0
        features['traj_max_step_distance'] = 0
        features['traj_distance_std'] = 0
        features['traj_straightness'] = 0
    
    # Angular features (direction changes)
    if len(coords) > 2:
        angles = []
        for i in range(len(coords) - 2):
            v1 = coords_array[i+1] - coords_array[i]
            v2 = coords_array[i+2] - coords_array[i+1]
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angles.append(np.abs(angle))
        
        features['traj_avg_angle_change'] = np.mean(angles)
        features['traj_max_angle_change'] = np.max(angles)
        features['traj_angle_std'] = np.std(angles)
    else:
        features['traj_avg_angle_change'] = 0
        features['traj_max_angle_change'] = 0
        features['traj_angle_std'] = 0
    
    return features


def extract_temporal_features(row):
    """Extract temporal features from timestamps."""
    features = {}
    
    try:
        start_time = pd.to_datetime(row['timestamp_start_radar_utc'])
        end_time = pd.to_datetime(row['timestamp_end_radar_utc'])
        
        # Duration
        duration = (end_time - start_time).total_seconds()
        features['duration_seconds'] = duration
        
        # Time of day features
        features['hour_of_day'] = start_time.hour
        features['minute_of_hour'] = start_time.minute
        features['day_of_week'] = start_time.dayofweek
        features['month'] = start_time.month
        
        # Cyclical encoding for hour
        features['hour_sin'] = np.sin(2 * np.pi * start_time.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * start_time.hour / 24)
        
        # Day period
        if 5 <= start_time.hour < 12:
            features['is_morning'] = 1
        elif 12 <= start_time.hour < 18:
            features['is_afternoon'] = 1
        elif 18 <= start_time.hour < 22:
            features['is_evening'] = 1
        else:
            features['is_night'] = 1
            
    except:
        features['duration_seconds'] = 0
        features['hour_of_day'] = 0
        features['minute_of_hour'] = 0
        features['day_of_week'] = 0
        features['month'] = 0
        features['hour_sin'] = 0
        features['hour_cos'] = 0
        features['is_morning'] = 0
        features['is_afternoon'] = 0
        features['is_evening'] = 0
        features['is_night'] = 0
    
    return features


def extract_radar_features(row):
    """Extract features from radar measurements."""
    features = {
        'radar_bird_size': row.get('radar_bird_size', 0),
        'airspeed': row.get('airspeed', 0),
        'min_z': row.get('min_z', 0),
        'max_z': row.get('max_z', 0),
    }
    
    # Altitude range
    features['altitude_range'] = features['max_z'] - features['min_z']
    
    # Speed category
    if features['airspeed'] < 10:
        features['speed_category'] = 0  # slow
    elif features['airspeed'] < 20:
        features['speed_category'] = 1  # medium
    else:
        features['speed_category'] = 2  # fast
    
    return features


def extract_all_features(df):
    """Extract all features from the dataframe."""
    print("Extracting features...")
    
    all_features = []
    
    for idx, row in df.iterrows():
        if idx % 500 == 0:
            print(f"Processing row {idx}/{len(df)}")
        
        features = {'track_id': row['track_id']}
        
        # Parse trajectory
        coords = parse_trajectory(row.get('trajectory'))
        
        # Extract features from different sources
        traj_features = extract_trajectory_features(coords)
        temporal_features = extract_temporal_features(row)
        radar_features = extract_radar_features(row)
        
        # Combine all features
        features.update(traj_features)
        features.update(temporal_features)
        features.update(radar_features)
        
        all_features.append(features)
    
    feature_df = pd.DataFrame(all_features)
    
    # Fill any missing values
    feature_df = feature_df.fillna(0)
    
    print(f"Extracted {len(feature_df.columns)} features")
    return feature_df

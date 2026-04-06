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
    
    # Add kinematic features
    kinematic_features = extract_kinematic_features(coords_array)
    features.update(kinematic_features)
    
    return features


def extract_kinematic_features(coords_array):
    """
    Extract physics-based kinematic features from trajectory.
    Assumes trajectory points are time-ordered.
    """
    features = {}
    
    if len(coords_array) < 2:
        # Not enough points for kinematic analysis
        return {
            'velocity_mean': 0, 'velocity_max': 0, 'velocity_std': 0,
            'horizontal_velocity_mean': 0, 'vertical_velocity_mean': 0,
            'acceleration_mean': 0, 'acceleration_max': 0, 'acceleration_std': 0,
            'acceleration_variance': 0, 'is_flapping': 0, 'is_gliding': 0,
            'climb_rate_mean': 0, 'climb_rate_max': 0, 'descent_rate_max': 0,
            'turn_radius_mean': 0, 'path_curvature': 0, 'tortuosity': 0
        }
    
    # Calculate 3D velocities (assuming uniform time steps)
    displacements = np.diff(coords_array, axis=0)
    
    # 3D velocity magnitude
    velocities_3d = np.sqrt(np.sum(displacements**2, axis=1))
    features['velocity_mean'] = np.mean(velocities_3d)
    features['velocity_max'] = np.max(velocities_3d)
    features['velocity_std'] = np.std(velocities_3d)
    
    # Horizontal velocity (X-Y plane)
    if coords_array.shape[1] >= 2:
        horizontal_velocities = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
        features['horizontal_velocity_mean'] = np.mean(horizontal_velocities)
    else:
        features['horizontal_velocity_mean'] = 0
    
    # Vertical velocity (Z-axis - climb/descent rate)
    if coords_array.shape[1] > 2:
        vertical_velocities = displacements[:, 2]
        features['vertical_velocity_mean'] = np.mean(vertical_velocities)
        features['climb_rate_mean'] = np.mean(vertical_velocities[vertical_velocities > 0]) if np.any(vertical_velocities > 0) else 0
        features['climb_rate_max'] = np.max(vertical_velocities) if len(vertical_velocities) > 0 else 0
        features['descent_rate_max'] = np.abs(np.min(vertical_velocities)) if len(vertical_velocities) > 0 else 0
    else:
        features['vertical_velocity_mean'] = 0
        features['climb_rate_mean'] = 0
        features['climb_rate_max'] = 0
        features['descent_rate_max'] = 0
    
    # Acceleration features (change in velocity)
    if len(velocities_3d) > 1:
        accelerations = np.diff(velocities_3d)
        features['acceleration_mean'] = np.mean(np.abs(accelerations))
        features['acceleration_max'] = np.max(np.abs(accelerations))
        features['acceleration_std'] = np.std(accelerations)
        features['acceleration_variance'] = np.var(accelerations)
        
        # Flight pattern detection
        # High acceleration variance = flapping (e.g., Songbirds)
        # Low acceleration variance = gliding (e.g., Birds of Prey)
        acc_var_threshold = 0.1
        features['is_flapping'] = 1 if features['acceleration_variance'] > acc_var_threshold else 0
        features['is_gliding'] = 1 if features['acceleration_variance'] <= acc_var_threshold else 0
    else:
        features['acceleration_mean'] = 0
        features['acceleration_max'] = 0
        features['acceleration_std'] = 0
        features['acceleration_variance'] = 0
        features['is_flapping'] = 0
        features['is_gliding'] = 0
    
    # Turn radius and curvature (requires at least 3 points)
    if len(coords_array) >= 3:
        turn_radii = []
        for i in range(len(coords_array) - 2):
            p1, p2, p3 = coords_array[i:i+3, :2]  # Use X-Y coordinates only
            
            # Calculate turn radius using three points
            # Using circumradius formula
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            
            # Semi-perimeter
            s = (a + b + c) / 2
            
            # Area using Heron's formula
            area_sq = s * (s - a) * (s - b) * (s - c)
            
            if area_sq > 1e-10:  # Avoid division by zero
                area = np.sqrt(area_sq)
                radius = (a * b * c) / (4 * area + 1e-10)
                turn_radii.append(radius)
        
        if turn_radii:
            features['turn_radius_mean'] = np.mean(turn_radii)
            features['path_curvature'] = 1.0 / (np.mean(turn_radii) + 1e-6)  # Inverse of radius
        else:
            features['turn_radius_mean'] = 0
            features['path_curvature'] = 0
    else:
        features['turn_radius_mean'] = 0
        features['path_curvature'] = 0
    
    # Tortuosity (path complexity measure)
    # Ratio of total path length to straight-line distance
    total_path = np.sum(np.sqrt(np.sum(displacements[:, :2]**2, axis=1)))
    straight_dist = np.linalg.norm(coords_array[-1, :2] - coords_array[0, :2])
    features['tortuosity'] = total_path / (straight_dist + 1e-6)
    
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

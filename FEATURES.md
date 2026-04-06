# Feature Documentation

## Baseline Features (run_baseline.py)

Total features extracted: **14**

### Radar Signature Features (5)
1. `radar_bird_size` - Size of radar return signal
2. `airspeed` - Measured bird airspeed
3. `min_z` - Minimum altitude
4. `max_z` - Maximum altitude
5. `altitude_range` - Difference between max and min altitude

### Temporal Features (8)
6. `duration_seconds` - Track duration in seconds
7. `hour_of_day` - Hour (0-23)
8. `day_of_week` - Day (0-6, Monday=0)
9. `month` - Month (1-12)
10. `hour_sin` - Sine of hour (cyclical encoding)
11. `hour_cos` - Cosine of hour (cyclical encoding)
12. `trajectory_length` - Length of WKB hex string (proxy for trajectory complexity)
13. `n_birds_observed` - Number of birds in observation (train only)

Note: Cyclical encoding (sin/cos) allows the model to understand that hour 23 is close to hour 0.

## Advanced Features (run_pipeline.py)

Total features extracted: **40+**

### Spatial Trajectory Features (20+)
Extracted from WKB-encoded LineString coordinates:

**Basic Statistics:**
- `traj_length` - Number of points in trajectory
- `traj_x_mean`, `traj_x_std`, `traj_x_min`, `traj_x_max` - X-axis statistics
- `traj_y_mean`, `traj_y_std`, `traj_y_min`, `traj_y_max` - Y-axis statistics
- `traj_z_mean`, `traj_z_std` - Z-axis (altitude) statistics

**Distance and Movement:**
- `traj_total_distance` - Total path length
- `traj_avg_step_distance` - Average distance between consecutive points
- `traj_max_step_distance` - Maximum single step distance
- `traj_distance_std` - Standard deviation of step distances
- `traj_straightness` - Ratio of straight-line distance to total path (0-1)

**Directional Features:**
- `traj_avg_angle_change` - Average turning angle
- `traj_max_angle_change` - Maximum turning angle
- `traj_angle_std` - Variability in direction changes

**Interpretation:**
- High straightness → Direct flight (likely migrating birds)
- High angle changes → Erratic movement (songbirds, raptors hunting)
- Large step distances → Fast-moving birds (gulls, geese)

### Enhanced Temporal Features (4)
Time-of-day categorization:
- `is_morning` - 5:00-12:00
- `is_afternoon` - 12:00-18:00
- `is_evening` - 18:00-22:00
- `is_night` - 22:00-5:00

Different bird groups have distinct activity patterns.

### Radar Features (6)
Same as baseline + categorization:
- `speed_category` - 0 (slow: <10), 1 (medium: 10-20), 2 (fast: >20)

## Feature Importance

Based on baseline model (top features by importance):

1. **Gulls** - Dominated by `hour_of_day`, `trajectory_length`, `altitude_range`
2. **Songbirds** - Influenced by `airspeed`, `duration_seconds`, time features
3. **Clutter** - Strong signal from `radar_bird_size`, `altitude_range`
4. **Waders** - Temporal patterns (`hour_sin/cos`) and altitude features

## Feature Engineering Best Practices

### For This Dataset:
1. **Handle missing data**: Use `.fillna(0)` after `pd.to_numeric(..., errors='coerce')`
2. **Normalize cyclical features**: Use sin/cos for hour-of-day
3. **Create interaction features**: altitude_range = max_z - min_z
4. **Categorical encoding**: Time periods as binary indicators
5. **Trajectory complexity**: Use WKB string length as proxy when parsing is slow

### General Tips:
- Test features incrementally (add one at a time, check OOF score)
- Remove correlated features (check correlation matrix)
- Scale features if using distance-based models (not needed for tree-based)
- Document domain knowledge (e.g., why certain birds fly at specific times)

## Future Feature Ideas

1. **Weather integration**: Temperature, wind speed (if available)
2. **Seasonal features**: Migration patterns by month
3. **Geographic clustering**: Latitude/longitude from trajectory start/end
4. **Trajectory Fourier transform**: Frequency domain features
5. **Statistical moments**: Skewness, kurtosis of trajectory distributions
6. **Group features**: Average/max speed within bird group
7. **Interaction terms**: `airspeed × altitude_range`, `hour × month`

## Feature Extraction Performance

- **Baseline** (simple features): ~1 second for full dataset
- **Advanced** (with trajectory parsing): ~30 seconds per 500 rows (~3 minutes total)

For faster iteration during development, use a subset of training data.

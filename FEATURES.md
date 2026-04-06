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

Total features extracted: **50+**

### Kinematic Features (17 NEW - Physics-Based)

Extracted from trajectory motion analysis. These features leverage **first principles of kinematics** to characterize flight behavior.

#### Velocity Features
- `velocity_mean` - Average 3D velocity magnitude: $\bar{v} = \frac{1}{n}\sum_{i=1}^{n}\|\vec{r}_{i+1} - \vec{r}_i\|$
- `velocity_max` - Maximum instantaneous velocity
- `velocity_std` - Velocity variation (indicates speed changes)
- `horizontal_velocity_mean` - Average velocity in X-Y plane (ground speed)
- `vertical_velocity_mean` - Average Z-axis velocity (climb/descent)

#### Acceleration Features  
Acceleration is the rate of change of velocity: $\vec{a} = \frac{d\vec{v}}{dt}$

- `acceleration_mean` - Average acceleration magnitude
- `acceleration_max` - Maximum acceleration (rapid maneuvers)
- `acceleration_std` - Acceleration variation
- `acceleration_variance` - **Flight pattern discriminator**:
  - High variance → Flapping flight (e.g., Songbirds)
  - Low variance → Gliding flight (e.g., Birds of Prey)

#### Flight Pattern Classification
- `is_flapping` - Binary indicator for flapping flight (high acceleration variance)
- `is_gliding` - Binary indicator for gliding flight (low acceleration variance)

#### Climb Dynamics
- `climb_rate_mean` - Average upward velocity (when ascending)
- `climb_rate_max` - Maximum climb rate
- `descent_rate_max` - Maximum descent rate (absolute value)

#### Path Geometry
- `turn_radius_mean` - Average turn radius from three-point circumcircles
- `path_curvature` - Inverse of turn radius: $\kappa = \frac{1}{r}$ (sharper turns = higher curvature)
- `tortuosity` - Path complexity: $\tau = \frac{\text{total path length}}{\text{straight-line distance}}$
  - τ ≈ 1: Straight flight
  - τ > 2: Highly tortuous path

**Why these matter:** Different bird species have distinct kinematic signatures based on their anatomy and behavior. Birds of Prey glide efficiently with low acceleration variance, while Songbirds use rapid wing flapping with high acceleration variance.

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

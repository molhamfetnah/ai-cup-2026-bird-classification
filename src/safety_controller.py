"""
Turbine Safety Controller Module
Industrial logic layer for bird strike prevention system.

Interfaces ML bird classifier with wind turbine control systems.
Provides real-time threat assessment and control recommendations.
"""

import numpy as np
from typing import Dict, Tuple, List


class TurbineSafetyController:
    """
    Real-time bird strike prevention system for wind turbines.
    
    Evaluates classified bird species, altitude, and proximity to determine
    appropriate turbine control actions (shutdown, slow, or normal operation).
    
    Attributes:
        turbine_height: Height of turbine hub (meters)
        blade_radius: Turbine blade radius (meters)
        critical_radius: Distance threshold for high-threat species (meters)
        warning_radius: Distance threshold for medium-threat species (meters)
    """
    
    # Species risk classification based on size, flight patterns, and conservation status
    HIGH_IMPACT_SPECIES = ['Birds of Prey', 'Geese', 'Cormorants']
    MEDIUM_IMPACT_SPECIES = ['Ducks', 'Gulls', 'Waders']
    LOW_IMPACT_SPECIES = ['Songbirds', 'Pigeons']
    
    # Control action types
    ACTION_SHUTDOWN = "TURBINE_SHUTDOWN"
    ACTION_SLOW = "TURBINE_SLOW"
    ACTION_NORMAL = "NORMAL_OPERATION"
    
    def __init__(self, 
                 turbine_height: float = 100.0,
                 blade_radius: float = 50.0,
                 critical_radius: float = 500.0,
                 warning_radius: float = 1000.0):
        """
        Initialize turbine safety controller with turbine specifications.
        
        Args:
            turbine_height: Hub height in meters (default: 100m)
            blade_radius: Blade radius in meters (default: 50m)
            critical_radius: Distance for immediate action (default: 500m)
            warning_radius: Distance for preventive action (default: 1000m)
        """
        self.turbine_height = turbine_height
        self.blade_radius = blade_radius
        self.critical_radius = critical_radius
        self.warning_radius = warning_radius
        
        # Calculate turbine swept area boundaries
        self.min_altitude = turbine_height - blade_radius
        self.max_altitude = turbine_height + blade_radius
    
    def evaluate_threat(self, 
                        bird_class: str, 
                        altitude: float, 
                        distance: float,
                        velocity: float = 0.0) -> Tuple[str, Dict]:
        """
        Evaluate bird strike threat and determine control action.
        
        Args:
            bird_class: Predicted bird species/category
            altitude: Bird altitude in meters
            distance: Horizontal distance from turbine in meters
            velocity: Bird velocity in m/s (optional, for trajectory prediction)
        
        Returns:
            Tuple of (control_action, threat_details)
            - control_action: One of ACTION_SHUTDOWN, ACTION_SLOW, ACTION_NORMAL
            - threat_details: Dictionary with threat level, reasoning, and metrics
        """
        threat_details = {
            'bird_class': bird_class,
            'altitude': altitude,
            'distance': distance,
            'velocity': velocity,
            'risk_level': 'UNKNOWN',
            'reasoning': []
        }
        
        # Check if bird is in turbine swept area (altitude-wise)
        in_swept_area = self.min_altitude <= altitude <= self.max_altitude
        
        # Determine species risk category
        if bird_class in self.HIGH_IMPACT_SPECIES:
            species_risk = 'HIGH'
        elif bird_class in self.MEDIUM_IMPACT_SPECIES:
            species_risk = 'MEDIUM'
        elif bird_class in self.LOW_IMPACT_SPECIES:
            species_risk = 'LOW'
        else:
            species_risk = 'UNKNOWN'
        
        # Decision logic
        # Priority 1: High-risk species within critical radius and swept area
        if species_risk == 'HIGH' and distance < self.critical_radius and in_swept_area:
            threat_details['risk_level'] = 'CRITICAL'
            threat_details['reasoning'].append(f"High-impact species ({bird_class}) within critical radius")
            threat_details['reasoning'].append(f"Altitude in blade sweep zone ({altitude:.1f}m)")
            return self.ACTION_SHUTDOWN, threat_details
        
        # Priority 2: Any bird in swept area at close distance
        if distance < self.critical_radius / 2 and in_swept_area:
            threat_details['risk_level'] = 'HIGH'
            threat_details['reasoning'].append(f"Bird within immediate proximity ({distance:.1f}m)")
            threat_details['reasoning'].append("Altitude intersects blade path")
            return self.ACTION_SHUTDOWN, threat_details
        
        # Priority 3: High-risk species approaching
        if species_risk == 'HIGH' and distance < self.warning_radius:
            threat_details['risk_level'] = 'ELEVATED'
            threat_details['reasoning'].append(f"Protected species ({bird_class}) approaching")
            threat_details['reasoning'].append(f"Distance: {distance:.1f}m")
            
            if in_swept_area:
                threat_details['reasoning'].append("Altitude: DANGEROUS")
                return self.ACTION_SLOW, threat_details
            else:
                threat_details['reasoning'].append(f"Altitude: Safe ({altitude:.1f}m)")
                return self.ACTION_NORMAL, threat_details
        
        # Priority 4: Medium-risk species in swept area
        if species_risk == 'MEDIUM' and in_swept_area and distance < self.warning_radius:
            threat_details['risk_level'] = 'MODERATE'
            threat_details['reasoning'].append(f"Medium-risk species in blade zone")
            threat_details['reasoning'].append(f"Distance: {distance:.1f}m, Altitude: {altitude:.1f}m")
            return self.ACTION_SLOW, threat_details
        
        # Default: Normal operation
        threat_details['risk_level'] = 'LOW'
        threat_details['reasoning'].append("No immediate threat detected")
        threat_details['reasoning'].append(f"Species: {bird_class} ({species_risk} risk)")
        threat_details['reasoning'].append(f"Distance: {distance:.1f}m, Altitude: {altitude:.1f}m")
        return self.ACTION_NORMAL, threat_details
    
    def batch_evaluate(self, detections: List[Dict]) -> List[Tuple[str, Dict]]:
        """
        Evaluate multiple bird detections and return prioritized actions.
        
        Args:
            detections: List of detection dictionaries with keys:
                       'bird_class', 'altitude', 'distance', 'velocity'
        
        Returns:
            List of (action, details) tuples sorted by threat level
        """
        results = []
        for detection in detections:
            action, details = self.evaluate_threat(
                bird_class=detection['bird_class'],
                altitude=detection.get('altitude', 0),
                distance=detection.get('distance', float('inf')),
                velocity=detection.get('velocity', 0.0)
            )
            results.append((action, details))
        
        # Sort by threat level priority
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'ELEVATED': 2, 'MODERATE': 3, 'LOW': 4}
        results.sort(key=lambda x: priority_order.get(x[1]['risk_level'], 5))
        
        return results
    
    def get_overall_action(self, detections: List[Dict]) -> str:
        """
        Get single overall control action for multiple detections.
        Uses most conservative action (shutdown > slow > normal).
        
        Args:
            detections: List of bird detection dictionaries
        
        Returns:
            Single control action (ACTION_SHUTDOWN, ACTION_SLOW, or ACTION_NORMAL)
        """
        if not detections:
            return self.ACTION_NORMAL
        
        results = self.batch_evaluate(detections)
        actions = [action for action, _ in results]
        
        # Return most conservative action
        if self.ACTION_SHUTDOWN in actions:
            return self.ACTION_SHUTDOWN
        elif self.ACTION_SLOW in actions:
            return self.ACTION_SLOW
        else:
            return self.ACTION_NORMAL
    
    def format_report(self, action: str, details: Dict) -> str:
        """
        Format threat assessment as human-readable report.
        
        Args:
            action: Control action
            details: Threat details dictionary
        
        Returns:
            Formatted string report
        """
        report = f"""
{'='*60}
TURBINE SAFETY ALERT - {details['risk_level']}
{'='*60}

CONTROL ACTION: {action}

Bird Classification: {details['bird_class']}
Current Position:
  - Altitude: {details['altitude']:.1f}m
  - Distance: {details['distance']:.1f}m
  - Velocity: {details['velocity']:.1f}m/s

Turbine Parameters:
  - Hub Height: {self.turbine_height}m
  - Blade Sweep: {self.min_altitude:.1f}m - {self.max_altitude:.1f}m

Risk Assessment:
"""
        for reason in details['reasoning']:
            report += f"  • {reason}\n"
        
        report += f"\n{'='*60}\n"
        return report


# Example usage
if __name__ == "__main__":
    # Initialize controller for a typical 3MW turbine
    controller = TurbineSafetyController(
        turbine_height=100,
        blade_radius=50,
        critical_radius=500,
        warning_radius=1000
    )
    
    # Example detections
    test_cases = [
        {
            'name': 'Critical: Birds of Prey in sweep zone',
            'bird_class': 'Birds of Prey',
            'altitude': 95,
            'distance': 400,
            'velocity': 15.0
        },
        {
            'name': 'Warning: Geese approaching',
            'bird_class': 'Geese',
            'altitude': 120,
            'distance': 800,
            'velocity': 20.0
        },
        {
            'name': 'Safe: Songbirds at safe altitude',
            'bird_class': 'Songbirds',
            'altitude': 20,
            'distance': 600,
            'velocity': 10.0
        },
        {
            'name': 'Moderate: Gulls in sweep zone',
            'bird_class': 'Gulls',
            'altitude': 105,
            'distance': 900,
            'velocity': 12.0
        }
    ]
    
    print("\n" + "="*70)
    print("TURBINE SAFETY CONTROLLER - TEST SCENARIOS")
    print("="*70 + "\n")
    
    for case in test_cases:
        print(f"\nScenario: {case['name']}")
        print("-" * 70)
        
        action, details = controller.evaluate_threat(
            bird_class=case['bird_class'],
            altitude=case['altitude'],
            distance=case['distance'],
            velocity=case['velocity']
        )
        
        print(controller.format_report(action, details))
    
    # Batch evaluation
    print("\n" + "="*70)
    print("BATCH EVALUATION - Overall Action")
    print("="*70)
    overall_action = controller.get_overall_action(test_cases)
    print(f"\nWith {len(test_cases)} detections, recommended action: {overall_action}")
    print("\nRationale: Multiple threats detected, using most conservative action.\n")

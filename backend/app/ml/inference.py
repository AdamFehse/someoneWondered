"""
Transformer Inference and System Generation

Generates stellar system parameters from transformer model.
Converts quantized tokens back to orbital elements.
"""

import torch
import numpy as np
from typing import Dict
import json
from pathlib import Path

from .transformer import TransformerForGeneration


class SystemGenerator:
    """Generate stellar systems using transformer model"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize generator

        Args:
            model_path: Path to trained transformer weights
            device: 'cpu' or 'cuda' for local
        """
        self.device = device
        self.num_bins = 253
        self.offset = 3
        self.model = self._load_model(model_path)
        self.model.eval()

        # Load normalization stats
        root_dir = Path(__file__).resolve().parents[3]
        stats_path = root_dir / "training/data/processed/normalization_stats.json"
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)

    def _load_model(self, model_path: str) -> TransformerForGeneration:
        """Load model from checkpoint"""
        model = TransformerForGeneration()

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        return model.to(self.device)

    def dequantize(self, token: int) -> float:
        """Convert token [3, 255] to continuous value [0, 1]"""
        # Tokens 0-2 are special, data tokens start at 3
        if token < 3:
            return 0.0
        return (token - self.offset) / (self.num_bins - 1)

    def generate_system(
        self,
        central_mass: float,
        num_bodies: int,
        temperature: float,
        top_k: int,
    ) -> Dict:
        """
        Generate a stellar system

        Args:
            central_mass: Central star mass (normalized [0, 1])
            num_bodies: Number of planets to generate
            temperature: Sampling temperature (higher = more diverse)
            top_k: Keep top-k tokens for sampling

        Returns:
            system: Dict with central_mass and orbital_elements
        """
        # Create prompt: [START_TOKEN, central_mass, num_bodies]
        central_mass_token = int(central_mass * (self.num_bins - 1) + self.offset)
        num_bodies_token = num_bodies + self.offset

        prompt = torch.tensor([[0, central_mass_token, num_bodies_token]])

        # Generate tokens
        max_tokens = num_bodies * 5 + 2  # 5 params per planet + START + END
        generated = self.model.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=self.device,
        )

        # Parse generated tokens
        tokens = generated[0].cpu().numpy()

        # Skip START token
        idx = 1
        denorm_central_mass = self._denorm_mass(self.dequantize(tokens[idx]))
        idx += 1

        # Skip num_bodies token
        idx += 1

        # Parse planet parameters
        planets = []
        for _ in range(num_bodies):
            if idx + 5 > len(tokens):
                break

            planet_tokens = tokens[idx:idx+5]
            planet = self._parse_planet_tokens(planet_tokens)
            planets.append(planet)
            idx += 5

        # Simple stabilization: cap eccentricity and randomize phases
        for planet in planets:
            planet['semi_major_axis'] = max(planet['semi_major_axis'], 0.05)
            planet['eccentricity'] = min(planet['eccentricity'], 0.3)
            planet['mean_anomaly'] = float(np.random.uniform(0, 2 * np.pi))
            planet['long_asc_node'] = float(np.random.uniform(0, 2 * np.pi))

        # Space orbits and phases for multi-planet systems
        if len(planets) > 1:
            planets.sort(key=lambda p: p['semi_major_axis'])
            min_ratio = 1.3
            for i in range(1, len(planets)):
                prev = planets[i - 1]['semi_major_axis']
                if planets[i]['semi_major_axis'] < prev * min_ratio:
                    planets[i]['semi_major_axis'] = prev * min_ratio
            for i, planet in enumerate(planets):
                phase = 2 * np.pi * i / len(planets)
                planet['mean_anomaly'] = phase
                planet['long_asc_node'] = phase

        return {
            'central_mass': float(denorm_central_mass),
            'planets': planets,
        }

    def _parse_planet_tokens(self, tokens: np.ndarray) -> Dict:
        """Convert 7 tokens to planet orbital elements"""
        values = [self.dequantize(int(t)) for t in tokens]

        return {
            'mass': self._denorm_mass(values[0]),
            'semi_major_axis': self._denorm_semi_major_axis(values[1]),
            'eccentricity': self._denorm_eccentricity(values[2]),
            'inclination': self._denorm_angle(values[3]),
            'arg_periapsis': self._denorm_angle(values[4]),
            'long_asc_node': 0.0,
            'mean_anomaly': 0.0,
        }

    def _denorm_mass(self, norm_value: float) -> float:
        """Denormalize mass from [0, 1]"""
        log_mass = norm_value * (self.stats['mass_max'] - self.stats['mass_min']) + self.stats['mass_min']
        return 10.0 ** log_mass

    def _denorm_semi_major_axis(self, norm_value: float) -> float:
        """Denormalize semi-major axis from [0, 1]"""
        log_sma = norm_value * (self.stats['semi_axis_max'] - self.stats['semi_axis_min']) + self.stats['semi_axis_min']
        return 10.0 ** log_sma

    def _denorm_eccentricity(self, norm_value: float) -> float:
        """Denormalize eccentricity from [0, 1]"""
        ecc = norm_value * (self.stats['ecc_max'] - self.stats['ecc_min']) + self.stats['ecc_min']
        return np.clip(float(ecc), 0.0, 0.99)

    def _denorm_angle(self, norm_value: float) -> float:
        """Denormalize angle from [0, 1] to [0, 2π]"""
        return float(norm_value * 2 * np.pi)

if __name__ == "__main__":
    # Test generation
    generator = SystemGenerator(model_path='models/transformer_v1.pt', device='cpu')

    system = generator.generate_system(
        central_mass=0.5,
        num_bodies=3,
        temperature=0.8,
    )

    print("Generated system:")
    print(f"  Central mass: {system['central_mass']:.3f} M☉")
    print(f"  Number of planets: {len(system['planets'])}")
    for i, planet in enumerate(system['planets']):
        print(f"  Planet {i+1}:")
        print(f"    Mass: {planet['mass']:.6f} M☉")
        print(f"    Semi-major axis: {planet['semi_major_axis']:.3f} AU")
        print(f"    Eccentricity: {planet['eccentricity']:.3f}")

    is_valid, reason = generator.validate_system(system)
    print(f"  Valid: {is_valid} ({reason})")

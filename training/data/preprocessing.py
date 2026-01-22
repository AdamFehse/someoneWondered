"""
NASA Exoplanet Archive Data Preprocessing Pipeline

Feature pipeline:
1) Fetch rows of confirmed planets from NASA (one row per planet).
2) Group by system name and keep multi-planet systems.
3) Normalize masses/log-semi-major-axis and angles to [0, 1].
4) Quantize to 8-bit tokens with START/PAD/END markers.
5) Augment by adding small noise to angular tokens.
6) Save train.npz + normalization_stats.json.
"""

import numpy as np
import requests
import pandas as pd
from pathlib import Path
import json
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physical constants
G = 1.0  # Gravitational constant (normalized units)
AU_TO_M = 1.496e11
SOLAR_MASS_TO_KG = 1.989e30

class ExoplanetPreprocessor:
    def __init__(
        self,
        output_dir: str = "training/data/processed",
        augment_multiplier: int = 10,
        max_systems: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Quantization parameters
        self.num_bins = 253
        self.vocab_size = 256
        self.augment_multiplier = augment_multiplier
        self.max_systems = max_systems

    def fetch_nasa_data(self) -> pd.DataFrame:
        """Fetch confirmed exoplanets from NASA API"""
        logger.info("Fetching NASA exoplanet data...")

        url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        query = (
            "SELECT pl_name, st_mass, pl_bmasse, pl_orbsmax, pl_orbeccen, "
            "pl_orbincl, pl_orblper "
            "FROM ps "
            "WHERE pl_orbsmax IS NOT NULL "
            "AND st_mass IS NOT NULL "
            "AND pl_bmasse IS NOT NULL "
            "AND pl_orbeccen IS NOT NULL "
            "AND pl_orbincl IS NOT NULL "
            "AND pl_orblper IS NOT NULL"
        )


        params = {
            "query": query,
            "format": "csv"
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Parse CSV response
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        logger.info(f"Downloaded {len(df)} exoplanet records")
        return df

    def filter_multi_planet_systems(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Group planets by system and filter for stable multi-planet systems"""
        logger.info("Filtering for multi-planet systems...")

        # Extract system name (usually first part before 'b', 'c', 'd', etc.)
        df['system'] = df['pl_name'].str.extract(r'(.*)\s+[a-z]$')[0]
        df['system'] = df['system'].fillna(df['pl_name'])

        systems = {}

        for system_name, group in df.groupby('system'):
            # Filter for systems with 2+ planets
            if len(group) < 2:
                continue

            # Check for missing values
            if group.isnull().any().any():
                continue

            planets = []
            for _, row in group.iterrows():
                planets.append({
                    'mass': float(row['pl_bmasse']),  # Planet mass in Earth masses
                    'semi_major_axis': float(row['pl_orbsmax']),  # AU
                    'eccentricity': float(row['pl_orbeccen']),
                    'inclination': float(np.deg2rad(row['pl_orbincl'])),
                    'arg_periapsis': float(np.deg2rad(row['pl_orblper'])),
                                                        })

            systems[system_name] = {
                'central_mass': float(group.iloc[0]['st_mass']),
                'planets': planets
            }

        logger.info(f"Filtered to {len(systems)} multi-planet systems")
        return systems

    def normalize_parameters(self, systems: Dict) -> Dict:
        """Normalize orbital parameters to [0, 1] range"""
        logger.info("Normalizing parameters...")

        # Collect all values for statistics
        all_masses = []
        all_semi_axes = []
        all_eccentricities = []
        all_inclinations = []

        for system in systems.values():
            all_masses.append(system['central_mass'])
            for planet in system['planets']:
                all_masses.append(planet['mass'])
                all_semi_axes.append(planet['semi_major_axis'])
                all_eccentricities.append(planet['eccentricity'])
                all_inclinations.append(planet['inclination'])

        # Log scale for masses and semi-major axes
        log_masses = np.log10(np.array(all_masses) + 1e-6)
        log_semi_axes = np.log10(np.array(all_semi_axes) + 1e-6)

        mass_min, mass_max = log_masses.min(), log_masses.max()
        semi_axis_min, semi_axis_max = log_semi_axes.min(), log_semi_axes.max()
        ecc_min, ecc_max = min(all_eccentricities), max(all_eccentricities)
        inc_min, inc_max = min(all_inclinations), max(all_inclinations)

        normalized = {}

        for name, system in systems.items():
            central_mass_norm = (np.log10(system['central_mass'] + 1e-6) - mass_min) / (mass_max - mass_min)

            planets_norm = []
            for planet in system['planets']:
                p_norm = {
                    'mass': (np.log10(planet['mass'] + 1e-6) - mass_min) / (mass_max - mass_min),
                    'semi_major_axis': (np.log10(planet['semi_major_axis'] + 1e-6) - semi_axis_min) / (semi_axis_max - semi_axis_min),
                    'eccentricity': (planet['eccentricity'] - ecc_min) / (ecc_max - ecc_min + 1e-6),
                    'inclination': planet['inclination'] / (2 * np.pi),
                    'arg_periapsis': planet['arg_periapsis'] / (2 * np.pi),
                                                        }
                planets_norm.append(p_norm)

            normalized[name] = {
                'central_mass': np.clip(central_mass_norm, 0, 1),
                'planets': planets_norm
            }

        # Save normalization stats
        stats = {
            'mass_min': float(mass_min),
            'mass_max': float(mass_max),
            'semi_axis_min': float(semi_axis_min),
            'semi_axis_max': float(semi_axis_max),
            'ecc_min': float(ecc_min),
            'ecc_max': float(ecc_max),
            'inc_min': float(inc_min),
            'inc_max': float(inc_max),
        }

        with open(self.output_dir / 'normalization_stats.json', 'w') as f:
            json.dump(stats, f)

        logger.info("Parameters normalized and stats saved")
        return normalized, stats

    def quantize_parameters(self, value: float) -> int:
        """Quantize continuous value [0, 1] to discrete token [0, 252]"""
        return int(np.clip(value * (self.num_bins - 1), 0, self.num_bins - 1))

    def dequantize_parameters(self, token: int) -> float:
        """Dequantize discrete token back to continuous value"""
        return token / (self.num_bins - 1)

    def create_sequences(self, normalized: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Convert normalized systems to transformer sequences"""
        logger.info("Creating transformer sequences...")

        sequences = []
        metadata = []

        special_tokens = {
            'START': 0,
            'PAD': 1,
            'END': 2,
        }

        offset = len(special_tokens)

        for system_name, system in normalized.items():
            num_planets = len(system['planets'])

            # Sequence: [START_TOKEN, central_mass_token, num_planets,
            #            planet1_params..., planet2_params..., ...]
            sequence = [special_tokens['START']]

            # Central mass (1 token)
            sequence.append(self.quantize_parameters(system['central_mass']) + offset)

            # Number of planets (1 token, clamped to max 8)
            sequence.append(min(num_planets, 8) + offset)

            # Planet parameters (5 tokens per planet)
            for planet in system['planets']:
                sequence.append(self.quantize_parameters(planet['mass']) + offset)
                sequence.append(self.quantize_parameters(planet['semi_major_axis']) + offset)
                sequence.append(self.quantize_parameters(planet['eccentricity']) + offset)
                sequence.append(self.quantize_parameters(planet['inclination']) + offset)
                sequence.append(self.quantize_parameters(planet['arg_periapsis']) + offset)
                                
            sequence.append(special_tokens['END'])

            # Pad to max sequence length (max 8 planets * 5 params + 3 special tokens)
            max_length = 64
            sequence = sequence[:max_length]
            sequence += [special_tokens['PAD']] * (max_length - len(sequence))

            sequences.append(sequence)
            metadata.append({
                'system_name': system_name,
                'num_planets': num_planets,
                'central_mass': system['central_mass']
            })

        logger.info(f"Created {len(sequences)} sequences")
        return np.array(sequences, dtype=np.int32), metadata

    def augment_data(self, sequences: np.ndarray, multiplier: int = 10) -> np.ndarray:
        """Data augmentation: random 3D rotations (for orbital elements)"""
        logger.info(f"Augmenting data {multiplier}x...")

        augmented = [sequences]

        # For orbital elements, we can add random offsets to angles
        for _ in range(multiplier - 1):
            aug_seq = sequences.copy()

            # Randomly perturb angular parameters
            # Indices for angle parameters in token space
            for i in range(len(aug_seq)):
                for j in range(3, len(aug_seq[i]), 5):  # Start after central_mass, num_planets
                    if j + 4 < len(aug_seq[i]):
                        # Add small random noise to angles
                        noise = np.random.randint(-5, 6, size=2)
                        aug_seq[i, j+3:j+5] = np.clip(aug_seq[i, j+3:j+5] + noise, 0, 255)

            augmented.append(aug_seq)

        result = np.vstack(augmented)
        logger.info(f"Augmented to {len(result)} total sequences")
        return result

    def process(self):
        """Execute full pipeline"""
        logger.info("Starting exoplanet data preprocessing...")

        # Step 1: Fetch data
        df = self.fetch_nasa_data()

        # Step 2: Filter multi-planet systems
        systems = self.filter_multi_planet_systems(df)
        if self.max_systems and len(systems) > self.max_systems:
            systems = dict(list(systems.items())[:self.max_systems])

        # Step 3: Normalize
        normalized, stats = self.normalize_parameters(systems)

        # Step 4: Create sequences
        sequences, metadata = self.create_sequences(normalized)

        # Step 5: Augment
        sequences_aug = self.augment_data(sequences, multiplier=self.augment_multiplier)

        # Save outputs
        output_file = self.output_dir / 'train.npz'
        np.savez(
            output_file,
            sequences=sequences_aug,
            metadata=metadata,
        )

        logger.info(f"Saved {len(sequences_aug)} augmented sequences to {output_file}")
        logger.info(f"Dataset shape: {sequences_aug.shape}")

        return sequences_aug, metadata

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess NASA exoplanet data")
    parser.add_argument("--output-dir", type=str, default="training/data/processed")
    parser.add_argument("--augment-multiplier", type=int, default=10)
    parser.add_argument("--max-systems", type=int, default=0)
    parser.add_argument("--testing-mode", action="store_true", help="Use fast defaults for quick iteration")
    parser.add_argument("--profile", choices=["test1", "test2", "full"], default="full", help="Preset preprocessing profile")
    args = parser.parse_args()

    if args.testing_mode and args.profile == 'full':
        args.profile = 'test1'

    if args.profile == 'test1':
        args.augment_multiplier = 2
        if args.max_systems == 0:
            args.max_systems = 200
    elif args.profile == 'test2':
        args.augment_multiplier = 3
        if args.max_systems == 0:
            args.max_systems = 800

    preprocessor = ExoplanetPreprocessor(
        output_dir=args.output_dir,
        augment_multiplier=args.augment_multiplier,
        max_systems=args.max_systems,
    )
    sequences, metadata = preprocessor.process()
    print(f"✓ Preprocessing complete: {sequences.shape}")
    print(f"✓ Sample sequence (first 10 tokens): {sequences[0][:10]}")

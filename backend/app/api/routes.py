"""
API Routes for Stellar System Generation and Simulation

Endpoints:
- POST /api/generate - Generate and simulate stellar system
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import logging
import os
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Import modules
from ..ml.inference import SystemGenerator
from ..physics.nbody import NBodySimulator

router = APIRouter()

# Global state (initialized on first request)
_generator = None
_simulator = None


def get_generator():
    """Lazy-load system generator"""
    global _generator
    if _generator is None:
        model_path = os.getenv('MODEL_PATH', 'models/transformer_v1.pt')
        model_path = Path(model_path)
        if not model_path.is_absolute():
            root_dir = Path(__file__).resolve().parents[3]
            model_path = root_dir / model_path
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading transformer model from {model_path} on {device}")
        _generator = SystemGenerator(model_path=model_path, device=device)
    return _generator


def get_simulator():
    """Get N-body simulator"""
    global _simulator
    if _simulator is None:
        _simulator = NBodySimulator(G=1.0)
    return _simulator


# Request/Response Models
class GenerateRequest(BaseModel):
    central_mass: float
    num_bodies: int
    temperature: float
    top_k: int
    simulation_timesteps: int
    simulation_dt: float



class BodyState(BaseModel):
    mass: float
    position: List[float]
    velocity: List[float]


class GenerateResponse(BaseModel):
    central_mass: float
    bodies: List[BodyState]
    trajectory: List[List[List[float]]]
    orbital_elements: List[Dict[str, float]]


@router.post("/generate", response_model=GenerateResponse)
async def generate_system(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a stellar system and simulate its dynamics

    Process:
    1. Use transformer to generate orbital parameters
    2. Convert to Cartesian initial conditions
    3. Run N-body simulation
    4. Return trajectory
    """
    logger.info(f"Generating system: mass={request.central_mass}, bodies={request.num_bodies}")

    # Get components
    generator = get_generator()
    simulator = get_simulator()

    # Generate system
    testing_mode = os.getenv("TESTING_MODE", "0") == "1"
    if testing_mode:
        request.num_bodies = min(request.num_bodies, 3)
        request.simulation_timesteps = min(request.simulation_timesteps, 2000)
        request.simulation_dt = min(request.simulation_dt, 0.02)

    system = generator.generate_system(
        central_mass=request.central_mass,
        num_bodies=request.num_bodies,
        temperature=request.temperature,
        top_k=request.top_k,
    )

    logger.info(f"System generated: {len(system['planets'])} planets")

    # Run simulation
    orbital_elems = system['planets']
    traj, masses = simulator.simulate(
        central_mass=system['central_mass'],
        orbital_elements=orbital_elems,
        timesteps=request.simulation_timesteps,
        dt=request.simulation_dt,
        return_every=max(1, request.simulation_timesteps // 1000),  # Limit output size
    )

    logger.info(f"Simulation complete: {traj.shape}")

    # Convert to response format
    bodies_initial = []
    for i, mass in enumerate(masses):
        bodies_initial.append(BodyState(
            mass=float(mass),
            position=traj[0][i].tolist(),
            velocity=[0.0, 0.0, 0.0] if i == 0 else [0.0, 0.0, 0.0],  # Approximate
        ))

    # Convert trajectory to list format
    trajectory_list = traj.tolist()

    response = GenerateResponse(
        central_mass=float(system['central_mass']),
        bodies=bodies_initial,
        trajectory=trajectory_list,
        orbital_elements=system['planets'],
    )

    logger.info("Response ready")
    return response

"""
Configuration management for Dynamic Market Sent trading bot.
Centralizes all environment variables, trading parameters, and system constants.
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading operation modes."""
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"


class ExchangeType(Enum):
    """Supported cryptocurrency exchanges."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for genetic algorithm evolution."""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elite_size: int = 5
    tournament_size: int = 3
    fitness_threshold: float = 0.85
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.population_size <= 1000:
            raise ValueError("Population size must be between 1 and 1000")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        logger.info("Genetic algorithm configuration validated")


@dataclass
class NeuralNetworkConfig:
    """Configuration for neural network architecture."""
    input_size: int = 42  # Number of features
    hidden_layers: list = field(default_factory=lambda: [64, 32, 16])
    output_size: int = 3  # [buy, sell, hold]
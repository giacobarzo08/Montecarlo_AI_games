from numpy import int64, float64
import numpy as np
from typing import NoReturn, Tuple
from matplotlib import pyplot as plt
import matplotlib
import io
import base64
from multiprocessing import Pool, cpu_count
matplotlib.use('Agg')

class Roulette:
    def __init__(self, slots: int64 = 36) -> None:
        if not isinstance(slots, (int, int64)):
            raise TypeError("Number of slots must be an integer.")
        
        if slots < 1:
            raise ValueError("Number of slots must be at least 1.")
        
        self._slots = slots

    @property
    def slots(self) -> int64:
        return self._slots

    @slots.setter
    def slots(self, value: int64 = 36) -> NoReturn:
        if not isinstance(value, (int, int64)):
            raise TypeError("Number of slots must be an integer.")
        
        if value < 1:
            raise ValueError("Number of slots must be at least 1.")
        
        self._slots = value
    
    def spin(self) -> int64:
        return int64(np.random.randint(0, self.slots))
    
    def simulate_straight_up(self, spins: int64 = int64(1000),
                             bet: float64 = float64(1),
                             number: int64 = int64(1)) -> float64:

        if not isinstance(spins, (int, int64)):
            raise TypeError("Number of spins must be an integer.")
        
        if not isinstance(bet, (float, float64)):
            raise TypeError("Bet amount must be a float.")
        
        if not isinstance(number, (int, int64)):
            raise TypeError("Bet number must be an integer.")
        
        if spins < 1:
            raise ValueError("Number of spins must be at least 1.")
        
        if bet <= 0:
            raise ValueError("Bet amount must be positive.")
        
        if number < 0 or number >= self.slots:
            raise ValueError("Bet number must be between 0 and number of slots - 1.")
        
        # OTTIMIZZAZIONE: Vettorizzazione con NumPy
        # Genera tutti gli spin in una volta
        spins_results = np.random.randint(0, self.slots, size=spins)
        
        # Calcola le vincite: +35*bet quando vince, -bet sempre
        wins = np.where(spins_results == number, 35 * bet, 0)
        total_win = np.sum(wins) - (bet * spins)
        
        return float64(total_win)
    
    def simulate_red_black(self, spins: int64 = int64(1000),
                           bet: float64 = float64(1)) -> float64:
        
        if not isinstance(spins, (int, int64)):
            raise TypeError("Number of spins must be an integer.")
        
        if not isinstance(bet, (float, float64)):
            raise TypeError("Bet amount must be a float.")
        
        if spins < 1:
            raise ValueError("Number of spins must be at least 1.")
        
        if bet <= 0:
            raise ValueError("Bet amount must be positive.")
        
        # OTTIMIZZAZIONE: Vettorizzazione con NumPy
        spins_results = np.random.randint(0, self.slots, size=spins)
        
        # Pari vince, tutti pagano la puntata
        wins = np.where(spins_results % 2 == 0, bet, 0)
        total_win = np.sum(wins) - (bet * spins)
        
        return float64(total_win)
    
    # Funzione helper per multiprocessing
    @staticmethod
    def _run_straight_up_batch(args):
        """Esegue un batch di simulazioni straight up"""
        slots, spins, bet, number, n_sims = args
        roulette = Roulette(slots)
        results = []
        for _ in range(n_sims):
            results.append(roulette.simulate_straight_up(spins, bet, number))
        return results
    
    @staticmethod
    def _run_red_black_batch(args):
        """Esegue un batch di simulazioni red/black"""
        slots, spins, bet, n_sims = args
        roulette = Roulette(slots)
        results = []
        for _ in range(n_sims):
            results.append(roulette.simulate_red_black(spins, bet))
        return results
    
    def plot_straight_up_simulation(self,
                                    spins: int64 = 1000,
                                    bet: float64 = float64(1),
                                    number: int64 = int64(1),
                                    x: int = 20,
                                    use_multiprocessing: bool = True,
                                    n_processes: int = None) -> str:
        
        if not isinstance(bet, (float, float64)):
            raise TypeError("Bet amount must be a float.")
        
        if not isinstance(number, (int, int64)):
            raise TypeError("Bet number must be an integer.")
        
        if not isinstance(x, int):
            raise TypeError("Number of simulations must be an integer.")
        
        if not isinstance(spins, (int, int64)):
            raise TypeError("Number of spins must be an integer.")
        
        if bet <= 0:
            raise ValueError("Bet amount must be positive.")
        
        if number < 0 or number >= self.slots:
            raise ValueError("Bet number must be between 0 and number of slots - 1.")
        
        if x < 1:
            raise ValueError("Number of simulations must be at least 1.")
        
        if spins < 1:
            raise ValueError("Number of spins must be at least 1.")
        
        # OTTIMIZZAZIONE: Multiprocessing
        if use_multiprocessing and x > 50:
            if n_processes is None:
                n_processes = cpu_count()
            
            # Suddividi il lavoro tra i processi
            sims_per_process = x // n_processes
            remainder = x % n_processes
            
            tasks = []
            for i in range(n_processes):
                n_sims = sims_per_process + (1 if i < remainder else 0)
                if n_sims > 0:
                    tasks.append((self.slots, spins, bet, number, n_sims))
            
            # Esegui in parallelo
            with Pool(n_processes) as pool:
                batch_results = pool.map(self._run_straight_up_batch, tasks)
            
            # Combina i risultati
            all_wins = [win for batch in batch_results for win in batch]
        else:
            # Versione sequenziale (per poche simulazioni)
            all_wins = []
            for _ in range(x):
                win = self.simulate_straight_up(spins=spins, bet=bet, number=number)
                all_wins.append(win)
        
        # Calcola il totale cumulativo
        results = np.cumsum(all_wins)

        fig, ax = plt.subplots()
        ax.plot(range(1, x + 1), results)
        ax.set_xlabel('Number of Simulations')
        ax.set_ylabel('Total Winnings')
        ax.set_title('Straight Up Bet Simulation Results')
        ax.grid(True)
        
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)
        
        return img_base64
    
    def plot_red_black_simulation(self,
                                  spins: int64 = 1000,
                                  bet: float64 = float64(1),
                                  x: int = 20,
                                  use_multiprocessing: bool = True,
                                  n_processes: int = None) -> str: 
        
        if not isinstance(bet, (float, float64)):
            raise TypeError("Bet amount must be a float.")
        
        if not isinstance(x, int):
            raise TypeError("Number of simulations must be an integer.")
        
        if not isinstance(spins, (int, int64)):
            raise TypeError("Number of spins must be an integer.")
        
        if bet <= 0:
            raise ValueError("Bet amount must be positive.")
        
        if x < 1:
            raise ValueError("Number of simulations must be at least 1.")
        
        if spins < 1:
            raise ValueError("Number of spins must be at least 1.")
        
        # OTTIMIZZAZIONE: Multiprocessing
        if use_multiprocessing and x > 50:
            if n_processes is None:
                n_processes = cpu_count()
            
            sims_per_process = x // n_processes
            remainder = x % n_processes
            
            tasks = []
            for i in range(n_processes):
                n_sims = sims_per_process + (1 if i < remainder else 0)
                if n_sims > 0:
                    tasks.append((self.slots, spins, bet, n_sims))
            
            with Pool(n_processes) as pool:
                batch_results = pool.map(self._run_red_black_batch, tasks)
            
            all_wins = [win for batch in batch_results for win in batch]
        else:
            all_wins = []
            for _ in range(x):
                win = self.simulate_red_black(spins=spins, bet=bet)
                all_wins.append(win)
        
        results = np.cumsum(all_wins)

        fig, ax = plt.subplots()
        ax.plot(range(1, x + 1), results)
        ax.set_xlabel('Number of Simulations')
        ax.set_ylabel('Total Winnings')
        ax.set_title('Red/Black Bet Simulation Results')
        ax.grid(True)
        
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close(fig)
        
        return img_base64

    def __str__(self) -> str:
        return f"Roulette with {self.slots} slots"
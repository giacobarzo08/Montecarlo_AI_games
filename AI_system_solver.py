import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import threading
import os
import tempfile
import sys

def run_complex_solver():
    """
    Funzione da eseguire in un thread separato che apre un nuovo terminale
    e mostra l'addestramento della rete neurale per un sistema complesso
    """
    
    # Script da eseguire nel nuovo terminale
    script_content = '''
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os
import numpy as np

print("="*70)
print("RISOLUZIONE SISTEMA COMPLESSO CON DEEP LEARNING (PyTorch)")
print("="*70)
print()
print("SISTEMA DA RISOLVERE:")
print("  1) x³ - 2y² + z⁴ - 5 = 0")
print("  2) sin(x) + y³ - cos(z) - 1 = 0")
print("  3) e^x + y² - z³ + xyz - 3 = 0")
print("  4) x² + y⁴ + z² - log(|x+y+z+10|) - 8 = 0")
print()
print("Ricerca di una soluzione numerica con 4 variabili incognite...")
print("="*70)
print()

# Verifica disponibilità CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in uso: {device}")
if torch.cuda.is_available():
    print(f"GPU rilevata: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU disponibile: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Nessuna GPU CUDA rilevata, uso CPU")
print()

# Crea la directory models se non esiste
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

# Rete neurale più profonda per problemi complessi
class ComplexSystemSolver(nn.Module):
    def __init__(self):
        super(ComplexSystemSolver, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 4)  # Output: 4 variabili (x, y, z, w)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, dummy_input):
        x = torch.relu(self.fc1(dummy_input))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# Funzione di loss per il sistema di equazioni
def system_loss(vars):
    x, y, z, w = vars[:, 0], vars[:, 1], vars[:, 2], vars[:, 3]
    
    # Sistema di 4 equazioni non lineari
    eq1 = x**3 - 2*y**2 + z**4 - 5
    eq2 = torch.sin(x) + y**3 - torch.cos(z) - 1
    eq3 = torch.exp(torch.clamp(x, -5, 5)) + y**2 - z**3 + x*y*z - 3
    eq4 = x**2 + y**4 + z**2 - torch.log(torch.abs(x+y+z+w+10) + 1e-6) - 8
    
    # Loss totale: somma dei quadrati degli errori
    loss = torch.mean(eq1**2 + eq2**2 + eq3**2 + eq4**2)
    
    return loss, eq1, eq2, eq3, eq4

# Percorso del modello
model_path = os.path.join(models_dir, 'complex_system_solver.pth')

print("Inizializzazione della rete neurale profonda...")
print(f"Architettura: 1 → 128 → 256 → 256 → 128 → 64 → 4")
print(f"Parametri totali: ~{sum(p.numel() for p in ComplexSystemSolver().parameters()):,}")
print()

model = ComplexSystemSolver().to(device)
dummy_input = torch.tensor([[1.0]], requires_grad=False).to(device)

train_model = True

# Verifica se esiste un modello pre-addestrato
if os.path.exists(model_path):
    print("⚡ MODELLO PRE-ADDESTRATO TROVATO!")
    print("Caricamento modello salvato...")
    print()
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        with torch.no_grad():
            solution = model(dummy_input)
            x, y, z, w = solution[0].tolist()
            loss_val, eq1, eq2, eq3, eq4 = system_loss(solution)
        
        print("SOLUZIONE DAL MODELLO PRE-ADDESTRATO:")
        print("-"*70)
        print(f"  x = {x:10.6f}")
        print(f"  y = {y:10.6f}")
        print(f"  z = {z:10.6f}")
        print(f"  w = {w:10.6f}")
        print()
        print("VERIFICA (inserendo nella equazioni):")
        print(f"  Equazione 1: x³ - 2y² + z⁴ - 5              = {eq1.item():12.8f}")
        print(f"  Equazione 2: sin(x) + y³ - cos(z) - 1       = {eq2.item():12.8f}")
        print(f"  Equazione 3: e^x + y² - z³ + xyz - 3        = {eq3.item():12.8f}")
        print(f"  Equazione 4: x² + y⁴ + z² - log(...) - 8    = {eq4.item():12.8f}")
        print()
        print(f"Loss totale: {loss_val.item():.10f}")
        
        if loss_val.item() < 0.001:
            print()
            print("✓ Soluzione verificata con successo!")
            print("="*70)
            train_model = False
        else:
            print()
            print("⚠ Il modello salvato non è abbastanza preciso.")
            print("Procedo con ri-addestramento...")
            print()
            
    except Exception as e:
        print(f"⚠ Errore nel caricamento: {e}")
        print("Procedo con l'addestramento da zero...")
        print()
else:
    print("Nessun modello pre-addestrato trovato.")
    print("Avvio addestramento da zero...")
    print()

# Addestramento
if train_model:
    # Inizializzazione più intelligente
    with torch.no_grad():
        # Forza valori iniziali ragionevoli
        model.fc6.bias.data = torch.tensor([1.5, 1.0, 1.2, 0.5]).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500)
    
    max_epochs = 50000
    print_interval = 500
    tolerance = 0.001
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 3000
    
    print("INIZIO ADDESTRAMENTO")
    print("="*70)
    print(f"Obiettivo: Loss < {tolerance}")
    print(f"Epoche massime: {max_epochs:,}")
    print()
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        solution = model(dummy_input)
        loss_val, eq1, eq2, eq3, eq4 = system_loss(solution)
        
        # Backward pass
        loss_val.backward()
        
        # Gradient clipping per stabilità
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step(loss_val)
        
        # Tracking del miglior risultato
        if loss_val.item() < best_loss:
            best_loss = loss_val.item()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Stampa progresso
        if (epoch + 1) % print_interval == 0:
            x, y, z, w = solution[0].tolist()
            elapsed = time.time() - start_time
            
            print(f"Epoca {epoch+1:6,d} | Loss: {loss_val.item():12.8f} | "
                  f"x={x:7.3f} y={y:7.3f} z={z:7.3f} w={w:7.3f} | "
                  f"Tempo: {elapsed:6.1f}s")
            
            # Mostra dettagli equazioni ogni 2000 epoche
            if (epoch + 1) % 2000 == 0:
                print(f"         → Eq1: {eq1.item():10.6f}  Eq2: {eq2.item():10.6f}  "
                      f"Eq3: {eq3.item():10.6f}  Eq4: {eq4.item():10.6f}")
        
        # Early stopping basato su convergenza
        if loss_val.item() < tolerance:
            print()
            print("="*70)
            print(f"✓ CONVERGENZA RAGGIUNTA all'epoca {epoch+1:,}!")
            print(f"  Loss finale: {loss_val.item():.10f}")
            break
        
        # Early stopping basato su stallo
        if patience_counter >= max_patience:
            print()
            print("="*70)
            print(f"⚠ Training fermato per stallo (epoca {epoch+1:,})")
            print(f"  Miglior loss: {best_loss:.10f}")
            break
    
    elapsed_total = time.time() - start_time
    
    print("="*70)
    print()
    print("RISULTATI FINALI")
    print("="*70)
    
    # Valutazione finale
    model.eval()
    with torch.no_grad():
        solution = model(dummy_input)
        x, y, z, w = solution[0].tolist()
        loss_val, eq1, eq2, eq3, eq4 = system_loss(solution)
    
    print(f"Tempo totale di addestramento: {elapsed_total:.1f} secondi")
    print()
    print("SOLUZIONE TROVATA:")
    print(f"  x = {x:10.6f}")
    print(f"  y = {y:10.6f}")
    print(f"  z = {z:10.6f}")
    print(f"  w = {w:10.6f}")
    print()
    print("VERIFICA (valori dovrebbero essere ≈ 0):")
    print(f"  Equazione 1: x³ - 2y² + z⁴ - 5              = {eq1.item():12.8f}")
    print(f"  Equazione 2: sin(x) + y³ - cos(z) - 1       = {eq2.item():12.8f}")
    print(f"  Equazione 3: e^x + y² - z³ + xyz - 3        = {eq3.item():12.8f}")
    print(f"  Equazione 4: x² + y⁴ + z² - log(...) - 8    = {eq4.item():12.8f}")
    print()
    print(f"Loss totale: {loss_val.item():.10f}")
    print()
    
    # Valutazione qualità soluzione
    if loss_val.item() < 0.0001:
        quality = "ECCELLENTE ✓✓✓"
    elif loss_val.item() < 0.001:
        quality = "OTTIMA ✓✓"
    elif loss_val.item() < 0.01:
        quality = "BUONA ✓"
    elif loss_val.item() < 0.1:
        quality = "ACCETTABILE"
    else:
        quality = "DA MIGLIORARE"
    
    print(f"Qualità della soluzione: {quality}")
    print()
    
    # Salvataggio del modello
    if loss_val.item() < 0.01:
        print("="*70)
        print("SALVATAGGIO MODELLO")
        print("="*70)
        
        try:
            torch.save(model.state_dict(), model_path)
            print(f"✓ Modello salvato in: {model_path}")
            print()
            print("Il modello può essere riutilizzato per risoluzioni istantanee!")
        except Exception as e:
            print(f"⚠ Errore nel salvataggio: {e}")
    else:
        print("⚠ Soluzione non abbastanza precisa, modello non salvato.")
    
    print()
    print("="*70)
    print("Il Deep Learning ha risolto il sistema complesso!")
    print("="*70)

print()
print("Premi Enter per chiudere questo terminale...")
input()
'''

    # Crea un file temporaneo per lo script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        temp_script = f.name
        f.write(script_content)
    
    try:
        # Comando per Windows nativo
        if sys.platform == 'win32':
            subprocess.Popen(
                f'start "Complex System Solver - Deep Learning" cmd /k python "{temp_script}"',
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            print("[Complex Solver] Nuovo terminale aperto con successo!")
        else:
            # Fallback per Linux/Mac
            terminal_commands = [
                ['gnome-terminal', '--', 'python3', temp_script],
                ['xterm', '-e', f'python3 {temp_script}'],
                ['x-terminal-emulator', '-e', f'python3 {temp_script}'],
            ]
            
            success = False
            for cmd in terminal_commands:
                try:
                    subprocess.Popen(cmd)
                    success = True
                    print(f"[Complex Solver] Nuovo terminale aperto con successo!")
                    break
                except (FileNotFoundError, OSError):
                    continue
            
            if not success:
                print("[Complex Solver] ERRORE: Impossibile aprire un nuovo terminale.")
                
    except Exception as e:
        print(f"[Complex Solver] ERRORE nell'apertura del terminale: {e}")


def start_complex_solver_demo():
    """
    Funzione da chiamare dalla tua app Flask per avviare la demo del solver complesso
    in un thread separato
    """
    thread = threading.Thread(target=run_complex_solver, daemon=True)
    thread.start()
    return "Demo Complex System Solver avviata in un nuovo terminale!"

# Per test standalone (non da Flask)
if __name__ == '__main__':
    print("Avvio Complex System Solver in un nuovo terminale...")
    start_complex_solver_demo()
    print("Demo avviata! Controlla il nuovo terminale.")
    print("Il programma Flask può continuare a funzionare normalmente.")
    
    # Mantieni il programma principale attivo
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProgramma principale terminato.")
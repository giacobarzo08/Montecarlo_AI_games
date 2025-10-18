import torch
import torch.nn as nn
import torch.optim as optim
import subprocess
import threading
import os
import tempfile
import sys

def run_pytorch_solver():
    """
    Funzione da eseguire in un thread separato che apre un nuovo terminale
    e mostra l'addestramento della rete neurale
    """
    
    # Script da eseguire nel nuovo terminale
    script_content = '''
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os

print("="*60)
print("RISOLUZIONE EQUAZIONE CON DEEP LEARNING (PyTorch)")
print("="*60)
print("Equazione: x^2 + 2x - 1 = 0")
print("="*60)
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
print(f"Directory modelli: {models_dir}")
print()

# Soluzioni analitiche per confronto
a, b, c = 1, 2, -1
delta = b**2 - 4*a*c
sol1 = (-b + math.sqrt(delta)) / (2*a)
sol2 = (-b - math.sqrt(delta)) / (2*a)
print(f"Soluzioni analitiche: x1 = {sol1:.6f}, x2 = {sol2:.6f}")
print()

# Rete neurale semplice
class EquationSolver(nn.Module):
    def __init__(self):
        super(EquationSolver, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Funzione di loss personalizzata per l'equazione x^2 + 2x - 1 = 0
def equation_loss(x_pred, target_direction=None):
    base_loss = torch.mean((x_pred**2 + 2*x_pred - 1)**2)
    
    # Penalizza soluzioni nella direzione sbagliata
    if target_direction == 'negative':
        # Penalizza fortemente valori positivi per model2
        penalty = torch.exp(torch.relu(x_pred)) * 0.1
        return base_loss + penalty
    elif target_direction == 'positive':
        # Penalizza fortemente valori negativi per model1
        penalty = torch.exp(torch.relu(-x_pred)) * 0.1
        return base_loss + penalty
    
    return base_loss

# Percorsi dei modelli
model1_path = os.path.join(models_dir, 'equation_solver_sol1.pth')
model2_path = os.path.join(models_dir, 'equation_solver_sol2.pth')

print("Inizializzazione delle reti neurali...")

# Inizializza i modelli
model1 = EquationSolver().to(device)
model2 = EquationSolver().to(device)

# Input diversi per trovare soluzioni diverse
input1 = torch.tensor([[1.0]], requires_grad=False).to(device)
input2 = torch.tensor([[-3.0]], requires_grad=False).to(device)

# IMPORTANTE: Forza l'inizializzazione del secondo modello verso valori negativi
# per evitare che converga alla soluzione positiva invece che a quella negativa
with torch.no_grad():
    initial_output = model2(input2).item()
    # Sposta l'output iniziale verso -3 (vicino alla soluzione negativa -2.414)
    adjustment = -3.0 - initial_output
    model2.fc3.bias.add_(adjustment)
    print(f"Output iniziale model2 prima: {initial_output:.4f}")
    print(f"Output iniziale model2 dopo aggiustamento: {model2(input2).item():.4f}")

train_models = True

# Verifica se esistono modelli pre-addestrati
if os.path.exists(model1_path) and os.path.exists(model2_path):
    print()
    print("⚡ MODELLI PRE-ADDESTRATI TROVATI!")
    print("Caricamento modelli salvati...")
    
    try:
        model1.load_state_dict(torch.load(model1_path, map_location=device, weights_only=True))
        model2.load_state_dict(torch.load(model2_path, map_location=device, weights_only=True))
        print("✓ Modelli caricati con successo!")
        print()
        
        # Valuta direttamente i modelli caricati
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            final_x1 = model1(input1).item()
            final_x2 = model2(input2).item()
        
        print("RISULTATI DAI MODELLI PRE-ADDESTRATI")
        print("-"*60)
        print(f"Soluzione 1 (Deep Learning): x = {final_x1:.6f}")
        print(f"Soluzione 1 (Analitica):     x = {sol1:.6f}")
        print(f"Errore assoluto:             {abs(final_x1 - sol1):.8f}")
        print()
        print(f"Soluzione 2 (Deep Learning): x = {final_x2:.6f}")
        print(f"Soluzione 2 (Analitica):     x = {sol2:.6f}")
        print(f"Errore assoluto:             {abs(final_x2 - sol2):.8f}")
        print()
        
        # Verifica
        verifica1 = final_x1**2 + 2*final_x1 - 1
        verifica2 = final_x2**2 + 2*final_x2 - 1
        
        print("VERIFICA (inserendo x nell'equazione):")
        print(f"x1: {final_x1:.6f}^2 + 2*{final_x1:.6f} - 1 = {verifica1:.8f}")
        print(f"x2: {final_x2:.6f}^2 + 2*{final_x2:.6f} - 1 = {verifica2:.8f}")
        print()
        print("="*60)
        print("Modelli pre-addestrati utilizzati con successo!")
        print("="*60)
        
        train_models = False
        
    except Exception as e:
        print(f"⚠ Errore nel caricamento: {e}")
        print("Procedo con l'addestramento da zero...")
        print()
        train_models = True
else:
    print("Nessun modello pre-addestrato trovato.")
    print("Avvio addestramento da zero...")
    print()

# Addestramento (solo se necessario)
if train_models:
    optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.01)
    
    max_epochs = 5000
    print_interval = 100
    tolerance = 1e-6  # Tolleranza per considerare la soluzione trovata
    
    print("RICERCA SOLUZIONE 1 (partendo da x ~ 1)")
    print("-"*60)
    
    for epoch in range(max_epochs):
        optimizer1.zero_grad()
        x1_pred = model1(input1)
        loss1 = equation_loss(x1_pred, target_direction='positive')
        loss1.backward()
        optimizer1.step()
        
        # Calcola errore rispetto alla soluzione analitica
        error1 = abs(x1_pred.item() - sol1)
        
        if (epoch + 1) % print_interval == 0:
            print(f"Epoca {epoch+1:4d} | x = {x1_pred.item():8.5f} | Loss = {loss1.item():.8f} | Errore = {error1:.8f}")
            time.sleep(0.05)
        
        # Early stopping: fermati quando l'errore è sufficientemente piccolo
        if error1 < tolerance:
            print(f"✓ Soluzione 1 trovata all'epoca {epoch+1}! Errore: {error1:.8f}")
            break
    
    print()
    print("RICERCA SOLUZIONE 2 (partendo da x ~ -3)")
    print("-"*60)
    
    for epoch in range(max_epochs):
        optimizer2.zero_grad()
        x2_pred = model2(input2)
        loss2 = equation_loss(x2_pred, target_direction='negative')
        loss2.backward()
        optimizer2.step()
        
        # Calcola errore rispetto alla soluzione analitica
        error2 = abs(x2_pred.item() - sol2)
        
        if (epoch + 1) % print_interval == 0:
            print(f"Epoca {epoch+1:4d} | x = {x2_pred.item():8.5f} | Loss = {loss2.item():.8f} | Errore = {error2:.8f}")
            time.sleep(0.05)
        
        # Early stopping
        if error2 < tolerance:
            print(f"✓ Soluzione 2 trovata all'epoca {epoch+1}! Errore: {error2:.8f}")
            break
    
    print()
    print("="*60)
    print("RISULTATI FINALI")
    print("="*60)
    
    # Predizioni finali
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        final_x1 = model1(input1).item()
        final_x2 = model2(input2).item()
    
    print(f"Soluzione 1 (Deep Learning): x = {final_x1:.6f}")
    print(f"Soluzione 1 (Analitica):     x = {sol1:.6f}")
    print(f"Errore assoluto:             {abs(final_x1 - sol1):.8f}")
    print()
    print(f"Soluzione 2 (Deep Learning): x = {final_x2:.6f}")
    print(f"Soluzione 2 (Analitica):     x = {sol2:.6f}")
    print(f"Errore assoluto:             {abs(final_x2 - sol2):.8f}")
    print()
    
    # Verifica
    verifica1 = final_x1**2 + 2*final_x1 - 1
    verifica2 = final_x2**2 + 2*final_x2 - 1
    
    print("VERIFICA (inserendo x nell'equazione):")
    print(f"x1: {final_x1:.6f}^2 + 2*{final_x1:.6f} - 1 = {verifica1:.8f}")
    print(f"x2: {final_x2:.6f}^2 + 2*{final_x2:.6f} - 1 = {verifica2:.8f}")
    print()
    
    # SALVATAGGIO DEI MODELLI
    print("="*60)
    print("SALVATAGGIO MODELLI")
    print("="*60)
    
    try:
        torch.save(model1.state_dict(), model1_path)
        torch.save(model2.state_dict(), model2_path)
        print(f"✓ Modello 1 salvato in: {model1_path}")
        print(f"✓ Modello 2 salvato in: {model2_path}")
        print()
        print("I modelli possono essere riutilizzati!")
        print("Copia la cartella 'models' per usarli su altri computer.")
    except Exception as e:
        print(f"⚠ Errore nel salvataggio: {e}")
    
    print()
    print("="*60)
    print("Il Deep Learning ha trovato le soluzioni!")
    print("="*60)

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
            # Usa 'start' per aprire un nuovo terminale cmd
            # /wait non è necessario perché vogliamo che torni subito
            subprocess.Popen(
                f'start "PyTorch Equation Solver" cmd /k python "{temp_script}"',
                shell=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            print("[PyTorch Solver] Nuovo terminale aperto con successo!")
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
                    print(f"[PyTorch Solver] Nuovo terminale aperto con successo!")
                    break
                except (FileNotFoundError, OSError):
                    continue
            
            if not success:
                print("[PyTorch Solver] ERRORE: Impossibile aprire un nuovo terminale.")
                
    except Exception as e:
        print(f"[PyTorch Solver] ERRORE nell'apertura del terminale: {e}")


def start_pytorch_demo():
    """
    Funzione da chiamare dalla tua app Flask per avviare la demo PyTorch
    in un thread separato
    """
    thread = threading.Thread(target=run_pytorch_solver, daemon=True)
    thread.start()
    return "Demo PyTorch avviata in un nuovo terminale!"

# Per test standalone (non da Flask)
if __name__ == '__main__':
    print("Avvio demo PyTorch in un nuovo terminale...")
    start_pytorch_demo()
    print("Demo avviata! Controlla il nuovo terminale.")
    print("Il programma Flask può continuare a funzionare normalmente.")
    
    # Mantieni il programma principale attivo
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProgramma principale terminato.")
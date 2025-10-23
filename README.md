# Montecarlo_AI_games
Questo progetto contiene una batteria di due giochi (il simulatore di una roulette ed un risolutore di equezioni alimentato dall'intelligenza artificiale di `PyTorch`) che consentono all'utente di capire divertendosi in cosa consiste il metodo Montecarlo.

Per chiarezza, si precisa che questo progetto è stato pensato per l'aula di matematica dell'open day di una Liceo Scientifico.

# Requistiti di sistema
Il software è stato sviluppato per Windows e non è mai stato testato su una macchina nativa Linux. Il risolutore di equazioni non funziona in WSL, poiché è impossibile creare una nuova instanza del file `cmd.exe` e non esiste una konsole Linux vera e propria.

Anche se non strettamente necessaria, è **vivamente consigliata** una scheda video dotata di **GPU Nvidia RTX** e almeno **4GB di memoria VRAM**, poiché l'intelligenza artificiale basata su PyTorch sa sfrutta un hardware Nvidia può essere centinaia di volte più veloce rispeto all'uso su CPU.

Il programma frutta i tipi di dato `float64` e `int64`, quindi è vivamente consigliato un **porcessore a 64 bit**.

Oltre alle libreria di sistema, è necessario installare le librerie `NumPy`, `Flask`, `MatPlotLib` e `PyTorch`.
> Per quest'ultima libreria, seguire la quida qui sotto.

Il programma è stato testato con **Python 3.13.7**.

# Installazione e uso
> Si parte dal presupposto che sul computer sia già installata una versione di Python (e che sia inclusda nel path) adatta e `git`

## Creazione dell'ambiente virtuale
Si consiglia di eseguire questo programma in un ambiente virtuale isolato, per "interferenze" e possibili errori dovuti ad altri progetti. Pre creare l'ambiente virtuale seguire le istruzioni che seguono.

> Questa guida fa uso di Windows PowerShell, apribile selezionado "Terminale" dopo aver fatto click col tasto destro del mouse sulla finestra di Windows sulla barra delle applicazioni

```PowerShell
cd %USERPROFILE%\documents # o la directory in cui creare l'ambiente
git clone https://github.com/giacobarzo08/Montecarlo_AI_games.git
cd Montecarlo_AI_games
python.exe -m venv montecarlo # sostiture montecarlo con il nome del prorpio ambiente
.\montecarlo\scripts\activate
# ora si è all'interno dell'ambiente virtiule.
```

Per uscire dall'ambiente virtuale è possibile eseguire in PowerShell `deactivate`.

## Installazione dei requisiti
Se si dispone di una GPU Nvidia, è necessario scoprire la versione dei CUDA Cores per poter installare la versione di `PyTorch` corretta. Per farlo basta eseguire nel terminale `nvcc --version`. L'output dovrebbe somigliare al seguente.
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_13:58:20_Pacific_Daylight_Time_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```
Da notare che all'ultima riga appare la versione dei CUDA, `13.0` nel mio caso.
E' possibile visitare il [sito ufficiale di `PyTorch`](https://pytorch.org/get-started/locally/) per scoprire la versione di Torch da installare. Se non si dispone dei CUDA Cores, selezionare la versione CPU-Only.

> Da notare che il software di Nvidia per l'uso dei CUDA può raggiungere i 3.5GB.
> 
> Si cosiglia di aggiornare il driver grafico all'ultima versione tramite Nvidia App.

Per installare le dipendanze eseguire nel terminale i seguenti comandi:
```PowerShell
pip install --no-cache-dir flask numpy matplotlib
# Incollare la riga di codice generata dal sito di PyTorch
```
## Uso
L'uso di questo prgramma è molto semplice ed è costituito da soli due passaggi:
1) eseguire il file `main.py` digitando nel terminale `python main.py`
2) visitare il server creato e seguire le indicazioni, per farlo bisogna aprire un browser (noi abbiamo testato su un borwser Chromium-like) e cercare nella barra dell'URL `LocalHost:5000`.

# Licenza ed uso
Come dal licenza allegata, il progetto è coperto da licenza MIT. La scritta di copyright nel file `index.html` ha il solo scopo di "fare scena" durante l'open day e non ha alcun valore.

Il progetto è stao scritto all'interno dell'Istituto di Istruzione Superiore "[Jean Monnet](https://www.ismonnet.edu.it/)"

# Uso su Linux
> Stiamo ancora testando il porgramma su Linux

Installazione delle dimendenze.
```bash
sudo apt install git zip unzip make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev libncurses5-dev libffi-dev libffi8 ubuntu-dev-tools
```
```bash
curl -fsSL https://pyenv.run | bash
```

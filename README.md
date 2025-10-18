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

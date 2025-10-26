from flask import Flask, render_template, request
from roulette import Roulette
from numpy import int64, float64
from AI_equation_solver import start_pytorch_demo
from AI_system_solver import start_complex_solver_demo
import traceback
import base64
import sys
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/roulette')
def roulette():
    return render_template('roulette.html')

@app.route('/AI_equation_solver')
def AI_equation_solver():
    return render_template('AI_equation_solver.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    bet_type = request.form.get('bet_type')
    spins = int64(request.form.get('spins', 1000))
    bet = float64(request.form.get('bet', 1.0))
    number = int64(request.form.get('number', 1))
    with_plot = request.form.get('with_plot') == 'true'
    simulations = int(request.form.get('simulations', 20))
    
    roulette = Roulette(slots=37)
    
    try:        
        if bet_type == 'straight_up':
            if with_plot:
                img64 = roulette.plot_straight_up_simulation(spins=spins, bet=bet, number=number, x=simulations)
                with open('static/plot.png', 'wb') as f:
                    f.write(base64.b64decode(img64))
                return render_template('plot.html', bet_type=bet_type)
            else:
                total_earnings = f"{roulette.simulate_straight_up(spins=spins, bet=bet, number=number):.2f}€".replace('.', ',')
                return render_template('result.html', total_earnings=total_earnings, bet_type=bet_type)
                
        elif bet_type == 'red_black':
            if with_plot:
                img64 = roulette.plot_red_black_simulation(spins=spins, bet=bet, x=simulations)
                with open('static/plot.png', 'wb') as f:
                    f.write(base64.b64decode(img64))
                return render_template('plot.html', bet_type=bet_type)
            else:
                total_earnings = f"{roulette.simulate_red_black(spins=spins, bet=bet):.2f}€".replace('.', ',')
                return render_template('result.html', total_earnings=total_earnings, bet_type=bet_type)
        
        else:
            raise ValueError("Tipo di scommessa non valido.")
        
    except Exception as e:
        traceback.print_exc()
        return f"Errore: {e}"

@app.route('/linear_solver', methods=['GET', 'POST'])
def linear_solver():
    type_equation = 'Un\'equazione lineare'
    try:
        start_pytorch_demo()
        return render_template('ai_result.html', type_equation=type_equation)
    except Exception as e:
        traceback.print_exc()
        return f"<p>Errore: {e}</p>"

@app.route('/system_solver', methods=['GET', 'POST'])
def complex_solver():
    type_equation = 'un sistema di equazioni non lineari'
    try:
        start_complex_solver_demo()
        return render_template('ai_result.html', type_equation=type_equation)
    except Exception as e:
        traceback.print_exc()
        return f"<p>Errore: {e}</p>"

@app.route('/remove_models', methods=['GET', 'POST'])
def remove_models():
    try:
        if sys.platform == 'win32':
            folder = os.path.join(os.path.expandvars('%LOCALAPPDATA%'), 'Temp', 'models')
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(folder)
            else:
                return "<pre>Errore durante la rimozione dei modelli: nessun modello da rimuovere.</pre>"
            return """
            <pre>Modelli rimossi con successo.</pre>
            <button onclick="window.location.href='/'">Torna alla pagina principale</button>
            """
        else:
            folder = '/tmp/models'
            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(folder)
            else:
                return "<pre>Errore durante la rimozione dei modelli: nessun modello da rimuovere.</pre>"
            return """
            <pre>Modelli rimossi con successo.</pre>
            <button onclick="window.location.href='/'">Torna alla pagina principale</button>
            """
    except Exception as e:
        traceback.print_exc()
        return f"<pre>Errore durante la rimozione dei modelli: {e}</pre>"

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)


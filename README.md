Predicción de Jugadas en Piedra, Papel o Tijeras usando Cadenas de Markov
=========================================================================

Este proyecto implementa un sistema de predicción de jugadas para el juego Piedra, Papel o Tijeras utilizando [cadenas de Markov](./Matemáticas%20de%20las%20cadenas%20de%20Markov.md). El sistema analiza el historial de jugadas del usuario para predecir su siguiente movimiento.

## Estructura del Proyecto

El proyecto está organizado en tres archivos principales:

- `model_rps_computed.py`: Implementa el modelo de cadena de Markov
- `probabilities_rps_computed.py`: Calcula las probabilidades de transición
- `file_handler.py`: Gestiona el almacenamiento y lectura del historial de jugadas

## Implementación

### Modelo de Markov (`model_rps_computed.py`)

El modelo utiliza la biblioteca `pomegranate` para implementar una cadena de Markov de orden 1. Las características principales son:

- Representación de jugadas:
  - Piedra (Rock) = 0
  - Papel (Paper) = 1
  - Tijeras (Scissors) = 2

- El modelo se inicializa con datos de entrenamiento básicos y se actualiza con el historial de jugadas del usuario.
- Utiliza tensores de PyTorch para el manejo eficiente de los datos.

### Análisis de Probabilidades (`probabilities_rps_computed.py`)

La clase `RPSProbabilityAnalyzer` proporciona métodos para analizar las probabilidades de transición:

- `transition_probability`: Calcula la probabilidad de transición entre dos jugadas
- `max_post_action_probability`: Obtiene la máxima probabilidad de transición desde una jugada
- `most_likely_post_action`: Predice la jugada más probable después de una jugada dada
- `most_likely_initial_action`: Determina la jugada inicial más probable

### Gestión de Datos (`file_handler.py`)

El `UserActionsFileHandler` maneja la persistencia de datos:

- Almacena el historial de jugadas en un archivo de texto
- Lee el historial para entrenar el modelo
- Maneja la creación del directorio y archivo si no existen

## Funcionamiento

1. El sistema se inicializa con datos de entrenamiento básicos
2. Se cargan las jugadas históricas del usuario
3. El modelo se entrena con estos datos
4. Se pueden realizar predicciones sobre:
   - La probabilidad de transición entre jugadas
   - La jugada más probable después de una jugada específica
   - La jugada inicial más probable

## Ejemplo de Uso

```python
analyzer = RPSProbabilityAnalyzer()

# Obtener probabilidad de transición
prob = analyzer.transition_probability(PAPER, PAPER)

# Obtener la jugada más probable después de Papel
next_action = analyzer.most_likely_post_action(PAPER)

# Obtener la probabilidad máxima después de una jugada
max_prob = analyzer.max_post_action_probability(PAPER)
```

## Instalación

Es necesario el uso del paquete [Pomegranate](https://pomegranate.readthedocs.io/en/stable/index.html):

> Pomegranate is a python package which implements fast, efficient, and extremely flexible probabilistic models ranging from probability distributions to Bayesian networks to mixtures of hidden Markov models.

`python -m venv venv`

`source venv/bin/activate`

Instalar la versión estricta de pomegranate `v1.0.4`:

`pip install -r requirements.txt`

o

`pip install pomegranate`

## Uso

```bash
cd rps
rps% python3 RPS_spock_lizard.py

Pick a choice (ROCK[0], PAPER[1], SCISSORS[2]): 1
Computer picked SCISSORS.
SCISSORS wins PAPER. You lost!

Another round? (y/n): y

Pick a choice (ROCK[0], PAPER[1], SCISSORS[2]): 0
Computer picked ROCK.
User and computer picked ROCK. Draw game!

Another round? (y/n): y

Pick a choice (ROCK[0], PAPER[1], SCISSORS[2]): 1
Computer picked ROCK.
PAPER wins ROCK. You win!

Another round? (y/n): n

['PAPER', 'PAPER', 'ROCK', 'SCISSORS', 'SCISSORS', 'SCISSORS', 'PAPER']
# historico de partidas guardadas en el fichero user_actions_history.txt
[1, 2, 0, 1, 2, 0]
...
```
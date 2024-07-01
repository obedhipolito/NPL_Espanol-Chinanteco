from pickle import load
from numpy import array
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# generar una secuencia de caracteres con un modelo de lenguaje
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
 # generar un número fijo de caracteres
    for _ in range(n_chars):
 # codificar los caracteres como enteros
        encoded = [mapping[char] for char in in_text]
 # cortar secuencias a una longitud fija
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
 # una codificacion caliente
        encoded = to_categorical(encoded, num_classes=len(mapping))
 # Predecir el carácter
        yhat = np.argmax(model.predict(encoded), axis=-1)
 # invierte un número entero del mapa a un carácter
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
  # añadir a la entrada
        in_text += char
    return in_text


# cargar el modelo
model = load_model('model.h5')
# cargar el mapeo
mapping = load(open('mapping.pkl', 'rb'))
# Poniendo a prueba el comienzo de la rima
print(generate_seq(model, mapping, 11, "Kià ñà sôī Mayo. atėī", 100))
# prueba en la línea media
print(generate_seq(model, mapping, 11, "Lateī maá'loò jm kie' ajäü, jö ajua", 100))
# prueba no en el original
print(generate_seq(model, mapping, 11, 'chie tiô', 100))
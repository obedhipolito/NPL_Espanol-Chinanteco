from numpy import array
from pickle import dump
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# cargar doc en la memoria
def load_doc(filename):
 # abrir archivo en modo solo lectura
 file = open(filename, 'r')
 # leer todo el texto
 text = file.read()
 # cerrar el archivo
 file.close()
 return text

# definir el modelo
def define_model(X):
 model = Sequential()
 model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
 model.add(Dense(vocab_size, activation='softmax'))
 # compilar modelo
 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 # resumen del modelo definido
 model.summary()
 plot_model(model, to_file='model.png', show_shapes=True)
 return model

# cargar
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')
# enteros codifican secuencias de caracteres
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
 # línea entera de codificación
 encoded_seq = [mapping[char] for char in line]
 # almacenar
 sequences.append(encoded_seq)
# tamaño del vocabulario
vocab_size = len(mapping)
print('Tamaño del vocabulario: %d' % vocab_size)
# separar en entrada y salida
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
# modelo adecuado
y = to_categorical(y, num_classes=vocab_size)
# definir modelo
model = define_model(X)
# Guardar el mapeo
model.fit(X, y, epochs=100, verbose=2)
# guardar el modelo en el archivo
model.save('model.h5')
dump(mapping, open('mapping.pkl', 'wb'))
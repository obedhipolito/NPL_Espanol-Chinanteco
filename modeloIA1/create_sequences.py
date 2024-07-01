# carga doc en la memoria
def load_doc(filename):
 # abrir el archivo como de sólo lectura
 file = open(filename, 'r')
 # leer todo el texto
 text = file.read()
 # cerrar el archivo
 file.close()
 return text
 
# guardar tokens en un archivo, un diálogo por línea
def save_doc(lines, filename):
 data = '\n'.join(lines)
 file = open(filename, 'w')
 file.write(data)
 file.close()
 
# cargar texto
raw_text = load_doc('../esp_chi.txt')
print(raw_text)
# Limpiar
tokens = raw_text.split()
raw_text = ' '.join(tokens)
# organizarse en secuencias de personajes
length = 40
sequences = list()
for i in range(length, len(raw_text)):
 # seleccionar secuencia de tokens
 seq = raw_text[i-length:i+1]
 # tienda
 sequences.append(seq)
print('Secuencias totales: %d' % len(sequences))
# guardar secuencias en un archivo
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)
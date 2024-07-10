import keras_nlp
import tensorflow as tf
import limpieza

ESP_VOCAB_SIZE = 15000
CHI_VOCAB_SIZE = 15000

# Función para entrenar el tokenizador
def train_word_piece(text_samples, vocab_size, reserved_tokens):
    word_piece_ds = tf.data.Dataset.from_tensor_slices(text_samples)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        #mantener de los datos de entrenamiento, batch es el tamaño de la muestra y prefetch es el número de muestras que se pueden cargar en memoria
        word_piece_ds.batch(1000).prefetch(4),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab

# Función para crear los vocabularios
def create_vocabs(train_pairs, reserved_tokens):
    esp_samples = [text_pair[0] for text_pair in train_pairs]
    esp_vocab = train_word_piece(esp_samples, ESP_VOCAB_SIZE, reserved_tokens)

    chi_samples = [text_pair[1] for text_pair in train_pairs]
    chi_vocab = train_word_piece(chi_samples, CHI_VOCAB_SIZE, reserved_tokens)

    return esp_vocab, chi_vocab

# Función para tokenizar
def tokenize_examples(esp_vocab, chi_vocab, text_pairs):
    #se asignan los tokenizadores a las variables esp_tokenizer y chi_tokenizer
    esp_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=esp_vocab, lowercase=False, strip_accents=False, split=True, split_on_cjk=False, suffix_indicator='##', oov_token='[UNK]')
    chi_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=chi_vocab, lowercase=False, strip_accents=False, split=True, split_on_cjk=False, suffix_indicator='##', oov_token='[UNK]', dtype='string')
    
	#realizar la tokenización de los ejemplos y mostrar los tokens y el texto recuperado después de la detokenización
    esp_input_ex = text_pairs[0][0]
    esp_tokens_ex = esp_tokenizer.tokenize(esp_input_ex)
    print("Spanish sentence: ", esp_input_ex)
    print("Tokens: ", esp_tokens_ex)
    print("Recovered text after detokenizing: ", esp_tokenizer.detokenize(esp_tokens_ex),)
     
    print()
    chi_input_ex = text_pairs[0][1]
    chi_tokens_ex = chi_tokenizer.tokenize(chi_input_ex)
    print("Chinanteco sentence: ", chi_input_ex)
    print("Tokens: ", chi_tokens_ex)
    print("Recovered text after detokenizing: ", chi_tokenizer.detokenize(chi_tokens_ex),)
    
    return esp_tokenizer, chi_tokenizer

def main():
    # Cargar los datos desde el módulo limpieza
    train_pairs, val_pairs, test_pairs = limpieza.main()
    
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    esp_vocab, chi_vocab = create_vocabs(train_pairs, reserved_tokens)
    
    print("Spanish Tokens: ", esp_vocab[100:110])
    print("Chinanteco Tokens: ", chi_vocab[100:110])
    
    tokenize_examples(esp_vocab, chi_vocab, train_pairs)

    return esp_vocab, chi_vocab

if __name__ == "__main__":
    main()

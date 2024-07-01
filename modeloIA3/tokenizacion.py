import keras_nlp
import tensorflow.data as tf_data
import limpieza

CHI_VOCAB_SIZE = 15000
SPA_VOCAB_SIZE = 15000

# Función para entrenar el tokenizador
def train_word_piece(text_samples, vocab_size, reserved_tokens):
    word_piece_ds = tf_data.Dataset.from_tensor_slices(text_samples)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        #mantener de los datos de entrenamiento, batch es el tamaño de la muestra y prefetch es el número de muestras que se pueden cargar en memoria
        word_piece_ds.batch(1000).prefetch(4),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab

# Función para crear los vocabularios
def create_vocabs(train_pairs, reserved_tokens):
    chi_samples = [text_pair[0] for text_pair in train_pairs]
    chi_vocab = train_word_piece(chi_samples, CHI_VOCAB_SIZE, reserved_tokens)

    spa_samples = [text_pair[1] for text_pair in train_pairs]
    spa_vocab = train_word_piece(spa_samples, SPA_VOCAB_SIZE, reserved_tokens)

    return chi_vocab, spa_vocab

# Función para tokenizar
def tokenize_examples(chi_vocab, spa_vocab, text_pairs):
    #configurar wordpiece tokenizers para traaajar con el chinanteco
    chi_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=chi_vocab, lowercase=False, strip_accents=False, split=True, special_tokens_in_strings=False)
    spa_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=spa_vocab, lowercase=False, strip_accents=False)
    
	#realizar la tokenización de los ejemplos y mostrar los tokens y el texto recuperado después de la detokenización
    chi_input_ex = text_pairs[0][0]
    chi_tokens_ex = chi_tokenizer.tokenize(chi_input_ex)
    print("Chinanteco sentence: ", chi_input_ex)
    print("Tokens: ", chi_tokens_ex)
    print("Recovered text after detokenizing: ", chi_tokenizer.detokenize(chi_tokens_ex),)
     
    print()
    spa_input_ex = text_pairs[0][1]
    spa_tokens_ex = spa_tokenizer.tokenize(spa_input_ex)
    print("Spanish sentence: ", spa_input_ex)
    print("Tokens: ", spa_tokens_ex)
    print("Recovered text after detokenizing: ", spa_tokenizer.detokenize(spa_tokens_ex),)
    
    return chi_tokenizer, spa_tokenizer

def main():
    # Cargar los datos desde el módulo limpieza
    train_pairs, val_pairs, test_pairs = limpieza.main()
    
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    chi_vocab, spa_vocab = create_vocabs(train_pairs, reserved_tokens)
    
    print("Chinanteco Tokens: ", chi_vocab[100:110])
    print("Spanish Tokens: ", spa_vocab[100:110])
    
    tokenize_examples(chi_vocab, spa_vocab, train_pairs)

    return chi_vocab, spa_vocab

if __name__ == "__main__":
    main()

import keras_nlp
import tensorflow.data as tf_data
import limpieza

ENG_VOCAB_SIZE = 15000
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
    eng_samples = [text_pair[0] for text_pair in train_pairs]
    eng_vocab = train_word_piece(eng_samples, ENG_VOCAB_SIZE, reserved_tokens)

    spa_samples = [text_pair[1] for text_pair in train_pairs]
    spa_vocab = train_word_piece(spa_samples, SPA_VOCAB_SIZE, reserved_tokens)

    return eng_vocab, spa_vocab

# Función para tokenizar
def tokenize_examples(eng_vocab, spa_vocab, text_pairs):
    eng_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=eng_vocab, lowercase=False)
    spa_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=spa_vocab, lowercase=False)
    
	#realizar la tokenización de los ejemplos y mostrar los tokens y el texto recuperado después de la detokenización
    eng_input_ex = text_pairs[0][0]
    eng_tokens_ex = eng_tokenizer.tokenize(eng_input_ex)
    print("English sentence: ", eng_input_ex)
    print("Tokens: ", eng_tokens_ex)
    print("Recovered text after detokenizing: ", eng_tokenizer.detokenize(eng_tokens_ex),)
     
    print()
    spa_input_ex = text_pairs[0][1]
    spa_tokens_ex = spa_tokenizer.tokenize(spa_input_ex)
    print("Spanish sentence: ", spa_input_ex)
    print("Tokens: ", spa_tokens_ex)
    print("Recovered text after detokenizing: ", spa_tokenizer.detokenize(spa_tokens_ex),)
    
    return eng_tokenizer, spa_tokenizer

def main():
    # Cargar los datos desde el módulo limpieza
    train_pairs, val_pairs, test_pairs = limpieza.main()
    
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    eng_vocab, spa_vocab = create_vocabs(train_pairs, reserved_tokens)
    
    print("English Tokens: ", eng_vocab[100:110])
    print("Spanish Tokens: ", spa_vocab[100:110])
    
    tokenize_examples(eng_vocab, spa_vocab, train_pairs)

    return eng_vocab, spa_vocab

if __name__ == "__main__":
    main()

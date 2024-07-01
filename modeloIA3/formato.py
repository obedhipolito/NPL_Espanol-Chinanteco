from keras import ops
import tensorflow.data as tf_data
import keras_nlp
import limpieza
import tokenizacion

MAX_SEQUENCE_LENGTH = 40
#numero de pasos
BATCH_SIZE = 64

chi_tokenizer = None
spa_tokenizer = None

def preprocess_batch(chi, spa):
    
    global chi_tokenizer, spa_tokenizer
    batch_size = ops.shape(spa)[0]

    chi = chi_tokenizer(chi)
    spa = spa_tokenizer(spa)

    # Pad `chi` to `MAX_SEQUENCE_LENGTH`.
    chi_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH,
        pad_value=chi_tokenizer.token_to_id("[PAD]"),
    )
    chi = chi_start_end_packer(chi)

    # Add special tokens (`"[START]"` and `"[END]"`) to `spa` and pad it as well.
    spa_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH + 1,
        start_value=spa_tokenizer.token_to_id("[START]"),
        end_value=spa_tokenizer.token_to_id("[END]"),
        pad_value=spa_tokenizer.token_to_id("[PAD]"),
    )
    spa = spa_start_end_packer(spa)

    return (
        {
            "encoder_inputs": chi,
            "decoder_inputs": spa[:, :-1],
        },
        spa[:, 1:],
    )


def make_dataset(pairs):
    chi_texts, spa_texts = zip(*pairs)
    chi_texts = list(chi_texts)
    spa_texts = list(spa_texts)
    dataset = tf_data.Dataset.from_tensor_slices((chi_texts, spa_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()

def main(train_pairs, val_pairs):
    global chi_tokenizer, spa_tokenizer
    
    # Obtener los vocabularios y tokenizadores usando las funciones de tokenizacion.py
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    chi_vocab, spa_vocab = tokenizacion.create_vocabs(train_pairs, reserved_tokens)
    chi_tokenizer, spa_tokenizer = tokenizacion.tokenize_examples(chi_vocab, spa_vocab, train_pairs)
    
    # No necesitamos volver a tokenizar aquí, solo crear los datasets
    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)
    
    # Mostrar un ejemplo del dataset procesado
    for inputs, targets in train_ds.take(1):
        print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
        print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
        print(f"targets.shape: {targets.shape}")
    
    return train_ds, val_ds

if __name__ == "__main__":
    train_pairs, val_pairs, _ = limpieza.main()
    main(train_pairs, val_pairs)
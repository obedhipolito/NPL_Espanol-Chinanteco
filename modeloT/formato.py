from keras import ops
import tensorflow.data as tf_data
import keras_nlp
import limpieza
import tokenizacion

MAX_SEQUENCE_LENGTH = 100
#numero de pasos
BATCH_SIZE = 40

esp_tokenizer = None
chi_tokenizer = None

def preprocess_batch(esp, chi):
    
    global esp_tokenizer, chi_tokenizer
    batch_size = ops.shape(chi)[0]

    esp = esp_tokenizer(esp)
    chi = chi_tokenizer(chi)

    # Pad `esp` to `MAX_SEQUENCE_LENGTH`.
    esp_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH,
        pad_value=esp_tokenizer.token_to_id("[PAD]"),
    )
    esp = esp_start_end_packer(esp)

    # Add special tokens (`"[START]"` and `"[END]"`) to `chi` and pad it as well.
    chi_start_end_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=MAX_SEQUENCE_LENGTH + 1,
        start_value=chi_tokenizer.token_to_id("[START]"),
        end_value=chi_tokenizer.token_to_id("[END]"),
        pad_value=chi_tokenizer.token_to_id("[PAD]"),
    )
    chi = chi_start_end_packer(chi)

    return (
        {
            "encoder_inputs": esp,
            "decoder_inputs": chi[:, :-1],
        },
        chi[:, 1:],
    )


def make_dataset(pairs):
    esp_texts, chi_texts = zip(*pairs)
    esp_texts = list(esp_texts)
    chi_texts = list(chi_texts)
    dataset = tf_data.Dataset.from_tensor_slices((esp_texts, chi_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()

def main(train_pairs, val_pairs):
    global esp_tokenizer, chi_tokenizer
    
    # Obtener los vocabularios y tokenizadores usando las funciones de tokenizacion.py
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    esp_vocab, chi_vocab = tokenizacion.create_vocabs(train_pairs, reserved_tokens)
    esp_tokenizer, chi_tokenizer = tokenizacion.tokenize_examples(esp_vocab, chi_vocab, train_pairs)
    
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
import keras_nlp
import keras
EPOCHS = 20 # This should be at least 10 for convergence
EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8
ESP_VOCAB_SIZE = 15000
CHI_VOCAB_SIZE = 15000
MAX_SEQUENCE_LENGTH = 100
#numero de pasos
BATCH_SIZE = 40

esp_tokenizer = None
chi_tokenizer = None

#construccion del modelo
def create_model():
    # Encoder
    encoder_inputs = keras.Input(shape=(None,), name="encoder_inputs")

    x = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=ESP_VOCAB_SIZE,
        sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBED_DIM,
    )(encoder_inputs)

    encoder_outputs = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
    )(inputs=x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)


    # Decoder
    decoder_inputs = keras.Input(shape=(None,), name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")

    x = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=CHI_VOCAB_SIZE,
        sequence_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBED_DIM,
    )(decoder_inputs)

    x = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS
    )(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
    x = keras.layers.Dropout(0.5)(x)
    decoder_outputs = keras.layers.Dense(CHI_VOCAB_SIZE, activation="softmax")(x)
    decoder = keras.Model(
        [
            decoder_inputs,
            encoded_seq_inputs,
        ],
        decoder_outputs,
    )
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])

    transformer = keras.Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs,
        name="transformer",
    )
    
    return transformer

def train_model(transformer, train_ds, val_ds):
    #Entrenamiento del modelo
    transformer.summary()
    transformer.compile(
        "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history = transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
    transformer.save("./transformer.h5")
    return history
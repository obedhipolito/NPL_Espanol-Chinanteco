import keras
import keras_nlp
from keras import ops
import random
import tokenizacion
import limpieza

# Asegúrate de que estas constantes coincidan con las usadas durante el entrenamiento
MAX_SEQUENCE_LENGTH = 40
ESP_VOCAB_SIZE = 15000
CHI_VOCAB_SIZE = 15000

def load_model_and_tokenizers():
    # Cargar el modelo
    transformer = keras.models.load_model("./transformer.h5")
    
    # Cargar los vocabularios
    train_pairs, _, _ = limpieza.load_and_prepare_data("../esp-chi.txt")
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    esp_vocab, chi_vocab = tokenizacion.create_vocabs(train_pairs, reserved_tokens)
    
    # Crear los tokenizadores
    esp_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=esp_vocab, lowercase=False)
    chi_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=chi_vocab, lowercase=False)
    
    return transformer, esp_tokenizer, chi_tokenizer

def decode_sequences(transformer, esp_tokenizer, chi_tokenizer, input_sentences):
    batch_size = 1

    # Tokenize the encoder input.
    encoder_input_tokens = ops.convert_to_tensor(esp_tokenizer(input_sentences))
    if len(encoder_input_tokens[0]) < MAX_SEQUENCE_LENGTH:
        pads = ops.full((1, MAX_SEQUENCE_LENGTH - len(encoder_input_tokens[0])), 0)
        encoder_input_tokens = ops.concatenate(
            [encoder_input_tokens.to_tensor(), pads], 1
        )

    # Define a function that outputs the next token's probability given the
    # input sequence.
    def next(prompt, cache, index):
        logits = transformer([encoder_input_tokens, prompt])[:, index - 1, :]
        # Ignore hidden states for now; only needed for contrastive search.
        hidden_states = None
        return logits, hidden_states, cache

    # Build a prompt of length 40 with a start token and padding tokens.
    length = 40
    start = ops.full((batch_size, 1), chi_tokenizer.token_to_id("[START]"))
    pad = ops.full((batch_size, length - 1), chi_tokenizer.token_to_id("[PAD]"))
    prompt = ops.concatenate((start, pad), axis=-1)

    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next,
        prompt,
        stop_token_ids=[chi_tokenizer.token_to_id("[END]")],
        index=1,  # Start sampling after start token.
    )
    generated_sentences = chi_tokenizer.detokenize(generated_tokens)
    return generated_sentences

def translate(transformer, esp_tokenizer, chi_tokenizer, input_sentence):
    print(f"Input sentence: {input_sentence}")
    
    # Tokenize input
    input_tokens = esp_tokenizer(input_sentence)
    print(f"Input tokens: {input_tokens}")
    
    translated = decode_sequences(transformer, esp_tokenizer, chi_tokenizer, [input_sentence])
    translated = translated.numpy()[0].decode("utf-8")
    print(f"Raw translation: {translated}")
    
    translated = (
        translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )
    
    # Tokenize output
    output_tokens = chi_tokenizer(translated)
    print(f"Output tokens: {output_tokens}")
    
    return translated

def main():
    transformer, esp_tokenizer, chi_tokenizer = load_model_and_tokenizers()
    
    while True:
        input_sentence = input("Ingrese una frase en inglés para traducir (o 'q' para salir): ")
        if input_sentence.lower() == 'q':
            break
        
        translated = translate(transformer, esp_tokenizer, chi_tokenizer, input_sentence)
        print(f"Traducción: {translated}\n")

if __name__ == "__main__":
    main()
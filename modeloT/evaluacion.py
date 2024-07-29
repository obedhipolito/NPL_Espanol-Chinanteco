import random
import keras
from keras import ops
import keras_nlp
import tokenizacion
import limpieza


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

#prueba cualitativa
def test_model_cualitative(transformer, esp_tokenizer, chi_tokenizer, test_pairs):
    test_esp_texts = [pair[0] for pair in test_pairs]
    for i in range(2):
        input_sentence = random.choice(test_esp_texts)
        translated = decode_sequences(transformer, esp_tokenizer, chi_tokenizer, [input_sentence])
        translated = translated.numpy()[0].decode("utf-8")
        translated = (
            translated.replace("[PAD]", "")
            .replace("[START]", "")
            .replace("[END]", "")
            .strip()
        )
        print(f"** Example {i} **")
        print(input_sentence)
        print(translated)
        print()
        
#prueba cuantitativa
def test_model_cuantitative(transformer, esp_tokenizer, chi_tokenizer, test_pairs):
    rouge_1 = keras_nlp.metrics.RougeN(order=1)
    rouge_2 = keras_nlp.metrics.RougeN(order=2)
    
    for test_pair in test_pairs[:30]:
        input_sentence = test_pair[0]
        reference_sentence = test_pair[1]

        translated_sentence = decode_sequences(transformer, esp_tokenizer, chi_tokenizer, [input_sentence])
        translated_sentence = translated_sentence.numpy()[0].decode("utf-8")
        translated_sentence = (
            translated_sentence.replace("[PAD]", "")
            .replace("[START]", "")
            .replace("[END]", "")
            .strip()
        )

        rouge_1(reference_sentence, translated_sentence)
        rouge_2(reference_sentence, translated_sentence)

    print("ROUGE-1 Score: ", rouge_1.result())
    print("ROUGE-2 Score: ", rouge_2.result())


def main():
    # Cargar el modelo
    transformer = keras.models.load_model("./transformer.h5")
    
    # Ejecutar el módulo limpieza
    train_pairs, _, test_pairs = limpieza.main()
    
    # Crear los tokenizadores
    reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
    esp_vocab, chi_vocab = tokenizacion.create_vocabs(train_pairs, reserved_tokens)
    esp_tokenizer, chi_tokenizer = tokenizacion.tokenize_examples(esp_vocab, chi_vocab, train_pairs)
    
    # Realizar pruebas de traducción cualitativas y cuantitativas
    print("Realizando pruebas cualitativas de traducción:")
    test_model_cualitative(transformer, esp_tokenizer, chi_tokenizer, test_pairs)
    
    print("Realizando pruebas cuantitativas de traducción:")
    test_model_cuantitative(transformer, esp_tokenizer, chi_tokenizer, test_pairs)

if __name__ == "__main__":
    main()
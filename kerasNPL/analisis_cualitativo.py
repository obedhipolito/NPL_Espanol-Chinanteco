import random
import keras
from keras import ops
import keras_nlp

MAX_SEQUENCE_LENGTH = 40

def decode_sequences(transformer, eng_tokenizer, spa_tokenizer, input_sentences):
    batch_size = 1

    # Tokenize the encoder input.
    encoder_input_tokens = ops.convert_to_tensor(eng_tokenizer(input_sentences))
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
    start = ops.full((batch_size, 1), spa_tokenizer.token_to_id("[START]"))
    pad = ops.full((batch_size, length - 1), spa_tokenizer.token_to_id("[PAD]"))
    prompt = ops.concatenate((start, pad), axis=-1)

    generated_tokens = keras_nlp.samplers.GreedySampler()(
        next,
        prompt,
        stop_token_ids=[spa_tokenizer.token_to_id("[END]")],
        index=1,  # Start sampling after start token.
    )
    generated_sentences = spa_tokenizer.detokenize(generated_tokens)
    return generated_sentences

def test_model(transformer, eng_tokenizer, spa_tokenizer, test_pairs):
    test_eng_texts = [pair[0] for pair in test_pairs]
    for i in range(2):
        input_sentence = random.choice(test_eng_texts)
        translated = decode_sequences([input_sentence])
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
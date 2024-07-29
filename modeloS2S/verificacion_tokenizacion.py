from matplotlib import ticker
import torch
import random
import matplotlib.pyplot as plt
from formato import get_dataloader, tensorFromSentence, EOS_token, batch_size, hidden_size, SOS_token
from modelo import EncoderRNN, AttnDecoderRNN
from entrenamiento import train
from limpieza import MAX_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.switch_backend('agg')

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence).to(device)
        input_length = input_tensor.size()[0]

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

        for di in range(MAX_LENGTH):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di] = decoder_attention.data
            _, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

    return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=4):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

input_lang, output_lang, train_dataloader, pairs = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

# Entrena el modelo (las épocas están en 80)
train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

encoder.eval()
decoder.eval()

evaluateRandomly(encoder, decoder)

# Función para mostrar la atención
def showAttention(input_sentence, output_words, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Configuración de los ejes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Mostrar etiquetas en cada tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(encoder, decoder, input_sentence, input_lang, output_lang)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

# Evaluación con atención
evaluateAndShowAttention('el niño va a la escuela')
evaluateAndShowAttention('los dueños de la cueva')
evaluateAndShowAttention('el león y el grillo')
evaluateAndShowAttention('dos señores fueron a cazar')

import torch
import limpieza
import modelo
from formato import get_dataloader, batch_size, hidden_size, device, SOS_token, EOS_token
from entrenamiento import train
#from evaluacion import evaluateRandomly

from modelo import EncoderRNN, AttnDecoderRNN

def main():
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

    #ejecutar con evaluacion
    #encoder.eval()
    #decoder.eval()
    #evaluateRandomly(encoder, decoder)

if __name__ == "__main__":
    main()
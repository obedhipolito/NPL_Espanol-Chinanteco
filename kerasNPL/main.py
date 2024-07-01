import limpieza
import tokenizacion
import formato
import modelo

def main():
    # Ejecutar el módulo limpieza
    train_pairs, val_pairs, test_pairs = limpieza.main()
    print("Limpieza de datos completada.")
    
    # Ejecutar el módulo tokenizacion
    eng_vocab, spa_vocab = tokenizacion.main()
    print("prueba de Tokenización completada.")

    train_ds, val_ds = formato.main(train_pairs, val_pairs)
    print("Preprocesamiento completado.")
    
    transformer = modelo.create_model()
    history = modelo.train_model(transformer, train_ds, val_ds)
    print("Entrenamiento del modelo completado.")
    
    eng_tokenizer, spa_tokenizer = tokenizacion.get_tokenizers(eng_vocab, spa_vocab)

    # Realizar pruebas de traducción
    print("Realizando pruebas de traducción:")
    modelo.test_model(transformer, eng_tokenizer, spa_tokenizer, test_pairs)

if __name__ == "__main__":
    main()

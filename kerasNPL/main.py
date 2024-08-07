import limpieza
import tokenizacion
import formato
import modelo
from  kerasNPL.evaluacion import test_model

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

if __name__ == "__main__":
    main()

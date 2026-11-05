import subprocess
import sys
from pathlib import Path

def main():
    # Elenco degli script nell'ordine esatto in cui devono essere eseguiti
    scripts = [
        "degradation.py", 
        "linear.py",
        "histogram.py", 
        "gamma.py",
        "evaluation.py"
    ]
    
    base_dir = Path(__file__).resolve().parent
    
    print("Avvio Image Enhancement")
    
    for script in scripts:
        script_path = base_dir / script
        
        if not script_path.exists():
            print(f"Errore: Il file {script} non è stato trovato in {base_dir}")
            sys.exit(1)
            
        print(f"\nEsecuzione di {script}")
        
        # Esegue lo script usando lo stesso interprete Python corrente
        result = subprocess.run([sys.executable, str(script_path)])
        
        # Se uno script fallisce, interrompo la pipeline
        if result.returncode != 0:
            print(f"\nErrore durante l'esecuzione di {script}. Pipeline interrotta.")
            sys.exit(result.returncode)
            
    print("\nPipeline completata con successo!")
    print("Tutte le immagini e il file metrics.csv sono stati generati.")

if __name__ == "__main__":
    main()
import runpy
import sys
from pathlib import Path
# Proje ana dizini 
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_pipeline import run_preprocessing, debug_sequence_builder
from src.experiments import run_batadal_seed_experiments, run_skab_seed_experiments

def main():
    print("=== 1. ADIM: Ön İşleme Başlıyor ===")
    run_preprocessing.main()
    
    PIPELINE_DIR = PROJECT_ROOT / "src" / "data_pipeline"
    
    print("\n=== 2. ADIM: ADASYN ile Veri Dengeleme Başlıyor ===")

    adasyn_path = PIPELINE_DIR / "preprocessing" / "balance_adasyn_dl.py"
    
    if not adasyn_path.exists():
        raise FileNotFoundError(f"Kritik Hata: 'balance_adasyn.py' şu konumda bulunamadı: {adasyn_path}")
        
    runpy.run_path(str(adasyn_path), run_name="__main__")
    
    print("\n=== 3. ADIM: Sequence (Zaman Penceresi) Oluşturma Başlıyor ===")

    sequence_path = PIPELINE_DIR / "run_sequence_building.py"
    
    if not sequence_path.exists():
        raise FileNotFoundError(f"Kritik Hata: 'run_sequence_building.py' şu konumda bulunamadı: {sequence_path}")
        
    runpy.run_path(str(sequence_path), run_name="__main__")
    
    print("\n=== 4. ADIM: Pipeline Doğrulama (Debug) ===")
    debug_sequence_builder.main()
    
    print("\n Tüm veri yükleme ve hazırlık işlemleri başarıyla tamamlandı!")

    # DEEP LEARNING (DERİN ÖĞRENME) DENEYLERİ
    print("\n" + "="*50 + "\n=== 2. BÖLÜM: DEEP LEARNING SEED DENEYLERİ ===\n" + "="*50)
    
    print("-> BATADAL Deep Learning Seed Deneyleri Başlıyor...")
    run_batadal_seed_experiments.main()
    
    print("\n-> SKAB Deep Learning Seed Deneyleri Başlıyor...")
    run_skab_seed_experiments.main()

if __name__ == "__main__":
    main()
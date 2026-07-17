from run_alergia_state_merging_experiment import (
    main as run_batadal_alergia,
)
from run_skab_alergia_state_merging_experiment import (
    main as run_skab_alergia,
)


def main():
    print("\n" + "=" * 72)
    print("ALERGIA-INSPIRED STATE-MERGING DENEYLERİ")
    print("=" * 72)

    print("\n[BATADAL DENEYİ BAŞLATILIYOR]")
    run_batadal_alergia()

    print("\n[SKAB DENEYİ BAŞLATILIYOR]")
    run_skab_alergia()

    print("\nTüm ALERGIA deneyleri tamamlandı.")


if __name__ == "__main__":
    main()
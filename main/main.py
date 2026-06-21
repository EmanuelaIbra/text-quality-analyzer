from writing_service import WritingService
from pages.text_redundancy_checker import print_report


def main():
    """
    Main entry point for the interactive command-line interface. Runs a continuous loop 
    accepting user text, processing it through the modular pipeline stages, and outputting 
    a step-by-step diagnostic breakdown.
    """
    print("Analizzatore e Riscrittore di Testi Italiani")
    print("--------------------------------------------")

    # Initialize the core writing framework pipeline
    service = WritingService()

    # Continuous terminal operational loop
    while True:
        # Accept text input dynamically via standard input stream
        text = input("\nInserisci il testo (oppure exit): ")

        # Break out of the execution loop immediately if the stop keyword is entered
        if text.lower() == "exit":
            break

        # Pass inputs to the backend engine using a fixed optimization style preset
        result = service.process(
    text,
    mode="concise",
    final_check=True,
    fast=False,
    include_full_analysis=True,
)

        # STAGE 1: Text Comparison Display
        print("\nOriginale:")
        print(result["original"])

        print("\nDopo Correzione Grammaticale:")
        print(result["grammar_corrected"])

        print("\nDopo Pulizia Ripetizioni Dirette:")
        print(result["polished"])

        # STAGE 2: Explicit Linguistic Error Tracking
        print("\nCorrezioni grammaticali trovate:")
        for match in result["grammar_matches"]:
            print(f"  - {match['message']}")
            # List structural fix suggestions if any exist
            if match["suggestions"]:
                print(f"     Suggerimenti: {match['suggestions']}")

        # STAGE 3: Structural Baseline Performance Analytics
        print("\nMetriche grammaticali prima della riscrittura:")
        for key, value in result["grammar_metrics_before_rewrite"].items():
            print(f"  {key}: {value}")

        # STAGE 4: Word-Level Repetition Lifecycle Comparison
        print("\nAnalisi Ripetizioni - Originale:")
        for key, value in result["repetition_original"].items():
            print(f"  {key}: {value}")

        print("\nAnalisi Ripetizioni - Testo Corretto:")
        for key, value in result["repetition_corrected"].items():
            print(f"  {key}: {value}")

        # STAGE 5: Semantic and Stylistic Redundancy Breakdown
        print("\nReport Ridondanza / Similarità:")
        print_report(result["redundancy_report"])

        # STAGE 6: Local GenAI Large Language Model Output
        print("\nTesto Riscritto da Ollama:")
        print(result["rewritten"])

        # STAGE 7: Final Structural Evaluation & Post-Checking Output
        print("\nTesto Finale Dopo Controllo Grammaticale:")
        print(result["final"])

        print("\nMetriche finali:")
        for key, value in result["final_metrics"].items():
            print(f"  {key}: {value}")

        print("\nAnalisi Ripetizioni - Finale:")
        for key, value in result["repetition_final"].items():
            print(f"  {key}: {value}")


# Conditional guard ensuring script execution logic maps strictly to active runtime environments
if __name__ == "__main__":
    main()
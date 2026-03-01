from main_pipeline import run_full_pipeline
from tests.bm25_test import run_bm25_test
from tests.bi_encoder_test import run_bi_encoder_test
from tests.cross_encoder_test import run_cross_encoder_test
from tests.hybrid_test import run_hybrid_test
from utils.data_loader import load_dataset

def get_dataset_size():
    try:
        dataset = load_dataset('data/climatefever.jsonl')
        return len(dataset)
    except:
        return 0

def main():
    dataset_size = get_dataset_size()
    
    while True:
        print("\nSelect an option:")
        print("1. Run full pipeline")
        print("2. BM25 test")
        print("3. Bi-Encoder test")
        print("4. Cross-Encoder test")
        print("5. Hybrid test")
        
        option = input("Enter option (1-5): ").strip()

        if option not in ['1', '2', '3', '4', '5']:
            print("Invalid option. Please try again.")
            continue
            
        if option == '1':
            claim_prompt = f"Enter claim index (0-{dataset_size-1}): "
        else:
            claim_prompt = f"Enter claim index (0-{dataset_size-1}, or -1 for all claims): "
            
        while True:
            claim_input = input(claim_prompt).strip()
            try:
                claim_index = int(claim_input)
                if claim_index == -1 and option != '1':
                    break
                if 0 <= claim_index < dataset_size:
                    break
                print(f"Index must be between 0 and {dataset_size-1} (or -1 for all claims if option 2-5).")
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
                continue

        if option == "1":
            run_full_pipeline(claim_index)
        elif option == "2":
            run_bm25_test(claim_index)
        elif option == "3":
            run_bi_encoder_test(claim_index)
        elif option == "4":
            run_cross_encoder_test(claim_index)
        elif option == "5":
            run_hybrid_test(claim_index)
            
        another = input("\nWould you like to run another test? (y/n): ").strip().lower()
        if another != 'y':
            break

if __name__ == "__main__":
    main()
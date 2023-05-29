import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="flan-t5/contradiction", help="Path, url or short name of the model")
    parser.add_argument("--saving_dir", type=str, default="promtped_flan-t5", help="Path for saving")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--meta_batch_size", type=int, default=1, help="Batch size for meta training")
    parser.add_argument("--dev_batch_size", type=int, default=8, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=557, help="Random seed")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="Batch size for validation")
    parser.add_argument("--max_history", type=int, default=2, help="max number of turns in the dialogue")
    parser.add_argument("--GPU", type=int, default=8, help="number of gpu to use")
    parser.add_argument("--model_name", type=str, default="bloom", help="use t5 or bert or gpt?")
    parser.add_argument("--data", type=str, default="eSNLI")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--label", type=str, default='contradiction')
    parser.add_argument("--auto_eval", action='store_true', default=False)

    args = parser.parse_args()
    print(args.auto_eval)
    return args
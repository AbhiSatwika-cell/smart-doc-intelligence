import torch
import transformers
import datasets
import sklearn
import pandas

print(f"PyTorch:       {torch.__version__}")
print(f"Transformers:  {transformers.__version__}")
print(f"Datasets:      {datasets.__version__}")
print(f"scikit-learn:  {sklearn.__version__}")
print(f"Pandas:        {pandas.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("\nAll imports successful. Ready to build.")
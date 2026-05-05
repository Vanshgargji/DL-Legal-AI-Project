# Model Pipeline Install Notes

Install these only when you are ready to run real transformer models.

```powershell
python -m pip install -r model_benchmark_outputs/requirements-models.txt
```

For GPU PyTorch, install the CUDA-specific wheel from the official PyTorch selector instead of the default CPU wheel.

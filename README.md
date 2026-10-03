# edge-quant-snapdragon

Quantizing a from-scratch decoder-only transformer for the Snapdragon NPU with Qualcomm AI Hub, measured on a real Samsung Galaxy S26 (Snapdragon 8 Elite Gen 5).

W8A8 runs 1.8x faster than FP16 and uses 3x less memory, but it breaks the model: top-1 agreement with FP32 drops to 25%. A local sensitivity sweep traces most of the damage to INT8 LayerNorm outputs and the residual stream, not to the Linear layers.

## Results

On-device, Galaxy S26, QNN runtime, fixed input of 64 tokens. Accuracy is measured on 32 held-out TinyStories validation samples.

| Variant | Latency | Peak memory | First load | Perplexity | Top-1 agree vs FP32 |
| --- | --- | --- | --- | --- | --- |
| PyTorch FP32 (CPU, reference) | - | - | - | 1134.7 | 100.0% |
| FP16 (NPU) | 2.5 ms | 260 MB | 3259 ms | 1136.1 | 99.5% |
| W8A16 | 2.4 ms | 93 MB | 1226 ms | 1849.3 | 60.9% |
| W8A8 | 1.4 ms | 88 MB | 993 ms | 6293.9 | 24.6% |

All three variants ran 100% on the NPU, with no CPU or GPU fallback. Results reproduced exactly across two runs.

- FP16 on the NPU is effectively lossless.
- Quantization cuts peak memory about 3x and cold-start load time 2.7-3.3x.
- Only W8A8 is meaningfully faster. W8A16 costs about the same compute as FP16.
- W8A8 loses most of the model's predictions. W8A16 loses less, but more than expected.

## Where INT8 hurts

To find which tensors cause the accuracy loss, `sensitivity.py` simulates quantization in PyTorch: symmetric fake-quant with per-channel INT8 weights and per-tensor activations calibrated on 200 training samples. It quantizes one part of the model at a time and measures perplexity on 200 validation samples.

| Quantized to INT8 | Perplexity vs FP32 | Top-1 agree |
| --- | --- | --- |
| All Linear weights and inputs | +0.5% | - |
| + output logits | +0.5% | 90.3% |
| + residual stream | +27% | 58.1% |
| + LayerNorm outputs | +129% | 50.0% |
| Same, with 16-bit activations | +0.4% | 91.7% |

- The Linear layers tolerate INT8 well. Weights and inputs together cost under 1%.
- LayerNorm outputs are the most sensitive tensor, followed by the residual stream. A few large channels set the per-tensor scale, and the small values round to zero. This is the same activation-outlier effect that motivates LLM.int8() and SmoothQuant.
- Quantizing the logits barely moves perplexity, but it flips about 10% of top-1 predictions, because this model's top candidates are close together.
- 16-bit activations remove the loss in simulation.

The simulation explains part of the on-device loss, not all of it. It reproduces +129% for W8A8 against +455% on the device. The remainder likely comes from ops it does not model: attention scores, softmax, and the requantization between ops. For W8A16 the device loses 63% where the simulation loses under 1%. That points to how the toolchain calibrates or places precision rather than to the 16-bit format itself, and it is the open question I would chase next.

## Setup

- Model: 6-layer decoder-only transformer built from scratch (d_model 512, FFN 2048, vocab 28,879, about 48M parameters), trained briefly on 200k TinyStories stories.
- Export: `torch.export` with a fixed `[1, 64]` int32 input and no KV cache, i.e. a single prefill pass.
- Quantization: AI Hub `submit_quantize_job`, INT8 weights with INT8 or INT16 activations, calibrated on 500 training stories.
- Compilation: `qnn_dlc` target runtime. Inputs and outputs stay float/int32 (no `--quantize_io`).
- Accuracy: perplexity over non-padding tokens, and top-1 agreement with the PyTorch FP32 model at each position.

## Limitations

- The model is small and lightly trained (FP32 perplexity around 1150), and 90-99% of its FFN activations are dead ReLUs. All results are relative to its own FP32 baseline. Absolute accuracy and the size of each effect would differ on a well-trained model.
- Prefill only, at a fixed 64-token length. There is no autoregressive decoding or KV cache on device.
- One device, and 32 on-device accuracy samples (each sample returns 7 MB of logits).
- Profiling uses random inputs. Latency does not depend on input values.

## Reproduce

```bash
pip install qai-hub torch datasets
qai-hub configure --api_token <TOKEN>

python export_model.py    # export and verify against PyTorch
python quantize.py        # AI Hub: quantize, compile and profile W8A8 and W8A16
python accuracy.py        # on-device inference, perplexity and top-1 agreement
python sensitivity.py     # local per-tensor quantization sweep
```

`data.py` holds the shared tokenization, calibration split and validation split.
import qai_hub as hub
from data import encode, get_splits

DEVICE_NAME = "Samsung Galaxy S26 (Family)" 
ONNX_MODEL_ID = "mqy5l4vvn"

client = hub.Client()
device = hub.Device(DEVICE_NAME)
onnx_model = hub.get_model(ONNX_MODEL_ID)

calib_sents, _ = get_splits()
calib = dict(tokens=[encode(s) for s in calib_sents])
print(f"calibration samples: {len(calib['tokens'])}, shape {calib['tokens'][0].shape}")
print("available dtypes:", [d.name for d in hub.QuantizeDtype])

configs = [
    ("w8a8",  hub.QuantizeDtype.INT8),
    ("w8a16", hub.QuantizeDtype.INT16),
]

results = {}
for name, act_dtype in configs:
    print(f"\n=== {name} ===")
    qjob = client.submit_quantize_job(
        model=onnx_model,
        calibration_data=calib,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=act_dtype,
        name=f"edge-lm-{name}",
    )
    cjob = client.submit_compile_job(
        model=qjob.get_target_model(),
        device=device,
        options="--target_runtime qnn_dlc",
        name=f"edge-lm-{name}-qnn",
    )
    pjob = client.submit_profile_job(
        model=cjob.get_target_model(),
        device=device,
        name=f"edge-lm-{name}-profile",
    )
    compiled_id = cjob.get_target_model().model_id
    results[name] = compiled_id
    print(f"{name}: compiled model {compiled_id}")
    print(f"{name}: profile {pjob.url}")
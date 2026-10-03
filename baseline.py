import torch, qai_hub as hub
from export_model import load, SEQ

client = hub.Client()
device = hub.Device("Samsung Galaxy S26 (Family)")
model = load()
x = torch.randint(0, 1000, (1, SEQ), dtype=torch.int32)
ep = torch.export.export(model, (x,))
specs = dict(tokens=((1, SEQ), "int32"))

onnx_job = client.submit_compile_job(model=ep, device=device, input_specs=specs,
                                     options="--target_runtime onnx")
fp_job = client.submit_compile_job(model=ep, device=device, input_specs=specs,
                                   options="--target_runtime qnn_dlc")
prof = client.submit_profile_job(model=fp_job.get_target_model(), device=device)

print("onnx model id:", onnx_job.get_target_model().model_id)
print("profile:", prof.url)
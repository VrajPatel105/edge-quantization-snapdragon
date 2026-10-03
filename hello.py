import torch, torchvision
import qai_hub as hub

client = hub.Client()  
device = hub.Device("Samsung Galaxy S26 (Family)")

model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").eval()
shape = (1, 3, 224, 224)
with torch.no_grad():
    exported = torch.export.export(model, (torch.rand(shape),))

compile_job = client.submit_compile_job(
    model=exported, device=device,
    input_specs=dict(image=shape),
    options="--target_runtime tflite",
)
profile_job = client.submit_profile_job(
    model=compile_job.get_target_model(), device=device,
)
print(profile_job.url)
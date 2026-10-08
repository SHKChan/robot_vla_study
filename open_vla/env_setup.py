import os

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


# --- Download the pretrained model ---
model = AutoModelForVision2Seq.from_pretrained(
    'openvla/openvla-7b',
    # bf16 keeps fp32's exponent range with fewer mantissa bits. fp16 has more
    # mantissa but overflows around 65504, which large transformer activations
    # can exceed. Requires Ampere (A100 / RTX 30xx) or newer.
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,   # stream weights in instead of doubling RAM
    trust_remote_code=True,   # OpenVLA ships custom modelling code
).cuda()
processor = AutoProcessor.from_pretrained('openvla/openvla-7b', trust_remote_code=True)


def predict(image: Image.Image, instruction: str):
    # This template is not cosmetic -- the model was trained on exactly this
    # wording, with a lowercase instruction and no trailing period.
    prompt = f'In: What action should the robot take to {instruction}?\nOut:'
    # The processor handles tokenization, image preprocessing and the BOS token.
    inputs = processor(prompt, image).to('cuda', dtype=torch.bfloat16)
    # unnorm_key selects WHICH dataset's percentile statistics to invert when
    # turning bin indices back into physical units. The model stores one set per
    # Open X-Embodiment dataset, so this cannot be left as None.
    return model.predict_action(**inputs, unnorm_key='bridge_orig', do_sample=False)  # greedy, deterministic


# --- Verify: smoke test with a dummy image ---
image = Image.new('RGB', (224, 224), color='red')
print(f'Predicted action: {predict(image, "pick up the red block")}')

# --- Verify: real scene ---
if os.path.exists('robot_scene.jpg'):
    # OpenVLA expects 224x224 RGB; convert() drops a PNG/RGBA alpha channel.
    image = Image.open('robot_scene.jpg').convert('RGB').resize((224, 224))
    action = predict(image, 'pick up the red block and place it in the blue bowl')
    # example: [0.1, -0.3, 0.5, 0.0, 0.0, 0.0, 1.0]
    #          [dx, dy, dz, droll, dpitch, dyaw, gripper]
    # The first six values are a RELATIVE end-effector displacement, to be
    # composed with the current pose. The seventh is an ABSOLUTE gripper command.
    # Treating the first six as an absolute pose sends the arm toward the origin.
    print(f'Action: {action}')

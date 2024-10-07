import time
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


def run_inference(model_name, prompt, server_url="localhost:8000", model_version="1"):
    triton_client = InferenceServerClient(url=server_url, verbose=False)
    prompt_np = np.array([prompt.encode("utf-8")], dtype=object)
    prompt_in = InferInput(name="PROMPT", shape=[1], datatype="BYTES")
    prompt_in.set_data_from_numpy(prompt_np, binary_data=True)
    results_out = InferRequestedOutput(name="RESULTS", binary_data=False)

    start_time = time.time()
    response = triton_client.infer(
        model_name=model_name,
        model_version=model_version,
        inputs=[prompt_in],
        outputs=[results_out],
    )
    elapsed_time = time.time() - start_time
    results = response.get_response()["outputs"][0]["data"][0]
    return results, elapsed_time

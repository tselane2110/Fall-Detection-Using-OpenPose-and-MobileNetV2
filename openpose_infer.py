import subprocess
import os
import shutil

def openpose_infer(image_path, out_path="result.png"):
    # Run OpenPose using subprocess
    cmd = [
        "./build/examples/openpose/openpose.bin",
        "--image_path", image_path,
        "--model_pose", "BODY_25",
        "--write_images", "temp_out/",
        "--write_json", "",
        "--display", "0",
        "--disable_blending", "false"
    ]

    subprocess.run(cmd, check=True)

    # Move rendered output image
    for f in os.listdir("temp_out"):
        if "rendered" in f:
            shutil.move(os.path.join("temp_out", f), out_path)
            shutil.rmtree("temp_out")
            break
    
    return out_path


"""
# Run it

from openpose_infer import openpose_infer

# Run inference
out = openpose_infer("input.jpg", "output.jpg")

from IPython.display import Image
Image(out)

"""
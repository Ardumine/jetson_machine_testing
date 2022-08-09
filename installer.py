import os
import shutil
print("Please wait...")
def create_dir (dir):
    try:
        os.system("mkdir "  + dir)
    except:
        pass

create_dir("/home/nvidia/nvdli-data")
create_dir("/home/nvidia/nvdli-data/classification")
create_dir("/home/nvidia/nvdli-data/machine")

files = os.listdir("src")
for file in files:
    print("cp src/" + file + " /home/nvidia/nvdli-data/machine")
    os.system("cp src/" + file + " /home/nvidia/nvdli-data/machine")

os.system("touch /home/nvidia/run_machine.sh")
open("/home/nvidia/run_machine.sh", "w").write("sudo docker run --expose 8080  --runtime nvidia -it --rm --network host     --volume ~/nvdli-data:/nvdli-nano/data     --device /dev/video0     nvcr.io/nvidia/dli/dli-nano-ai:v2.0.1-r32.6.1 python3 /nvdli-nano/data/machine/run.py")

os.system("touch /home/nvidia/jupiter.sh")
open("/home/nvidia/jupiter.sh", "w").write("sudo docker run --expose 8080  --runtime nvidia -it --rm --network host     --volume ~/nvdli-data:/nvdli-nano/data     --device /dev/video0     nvcr.io/nvidia/dli/dli-nano-ai:v2.0.1-r32.6.1")
os.system("chmod +x /home/nvidia/jupiter.sh")
os.system("chmod +x /home/nvidia/run_machine.sh")

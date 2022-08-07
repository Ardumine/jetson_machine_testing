import os
import shutil
print("Please wait...")
def create_dir (dir):
    try:
        os.system("mkdir "  + dir)
    except:
        pass

create_dir("~/nvdli-data")
create_dir("~/nvdli-data/classification")
create_dir("~/nvdli-data/machine")

files = os.listdir("src")
for file in files:
    print("cp src/" + file + " ~/nvdli-data/machine")
    os.system("cp src/" + file + " ~/nvdli-data/machine")

open("~/run_machine.sh", "w").write("sudo docker run --expose 8080  --runtime nvidia -it --rm --network host     --volume ~/nvdli-data:/nvdli-nano/data     --device /dev/video0     nvcr.io/nvidia/dli/dli-nano-ai:v2.0.1-r32.6.1 cd /nvdli-nano/data/machine/ && python3 main.py")

open("~/jupiter.sh", "w").write("sudo docker run --expose 8080  --runtime nvidia -it --rm --network host     --volume ~/nvdli-data:/nvdli-nano/data     --device /dev/video0     nvcr.io/nvidia/dli/dli-nano-ai:v2.0.1-r32.6.1")
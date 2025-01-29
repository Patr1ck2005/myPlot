import vtk
gpu_info = vtk.vtkGPUInfoList()
num_gpus = gpu_info.GetNumberOfGPUs()
print(f"Detected GPUs: {num_gpus}")

for i in range(num_gpus):
    gpu = gpu_info.GetGPUInfo(i)
    print(f"GPU {i}: {gpu.GetDescription()}")

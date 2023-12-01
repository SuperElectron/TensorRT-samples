from ultralytics import YOLO


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Incorrect usage")
        print(f"Usage:   python3 {sys.argv[0]} /path/to/model")
        print(f"Example: python3 {sys.argv[0]} ../models/yolov8m.pt")
        sys.exit()

    # Model can be downloaded from https://github.com/ultralytics/ultralytics
    model = YOLO(sys.argv[1])
    model.fuse()
    model.info(verbose=False)  # Print model information
    model.export(format="onnx", opset=12) # Export the model to onnx using opset 12
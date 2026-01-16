
import sys
import os
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.join(os.getcwd(), 'masaustu_segmentasyon'))

from mantik.ai_model import get_model, load_model

def test_analyze():
    print("Testing analyze method...")
    
    # Mock model loading - we might not have the .pth file available or valid path in this context
    # but we can check if the method exists and runs until model inference if model is None
    
    # Try to load model if path exists, otherwise we mock it for structure testing
    model_path = r"c:\Users\Mustafa\Desktop\yeni_app\masaustu_segmentasyon\calcaneus_ultimate_model.pth"
    
    try:
        if os.path.exists(model_path):
            load_model(model_path)
            model = get_model()
            print("Model loaded.")
        else:
            print(f"Model file not found at {model_path}. Testing structure only.")
            load_model(model_path) # multiple try/except blocks inside might handle it or fail
            model = get_model()
            
    except Exception as e:
        print(f"Model load failed (expected if file missing): {e}")
        return

    if model is None:
        print("Model is None, cannot test inference.")
        return

    # Create dummy image
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(dummy_img, (50, 50), 30, (255, 255, 255), -1)
    
    # Test numpy array input
    print("\nTest 1: Numpy Input")
    try:
        result = model.analyze(dummy_img)
        print("Result keys:", result.keys())
        if "mask" in result:
            print("Mask shape:", result["mask"].shape)
            print("Status:", result["status"])
        else:
            print("Error:", result.get("error"))
    except Exception as e:
        print(f"Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analyze()

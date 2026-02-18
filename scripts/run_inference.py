"""
Generate final inference predictions using the best trained model
Run this script to create the final_predictions folder with annotated test images
"""
import os
from ultralytics import YOLO

def run_inference():
    os.chdir('e:/Dota_yolo_project')
    
    print("\n" + "="*70)
    print(" RUNNING FINAL MODEL INFERENCE")
    print("="*70 + "\n")
    
    print("Loading best model: runs/obb_v24/weights/best.pt")
    model = YOLO('runs/obb_v24/weights/best.pt')
    
    print("\nRunning inference on test images...")
    print("  → Source: DOTA/test/images")
    print("  → Image Size: 1024×1024")
    print("  → Confidence Threshold: 0.25")
    print("  → Output Directory: runs/obb/final_predictions")
    print("\nThis may take several minutes depending on the number of images...\n")
    
    results = model.predict(
        source='DOTA/test/images',
        imgsz=1024,
        conf=0.25,
        project='runs/obb',
        name='final_predictions',
        save=True,
        verbose=True
    )
    
    print("\n" + "="*70)
    print(" ✅ INFERENCE COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: runs/obb/final_predictions/")
    print(f"Total images processed: {len(results)}")
    print("\n📁 Output Structure:")
    print("   runs/obb/final_predictions/")
    print("   ├── images/")
    print("   │   ├── image_with_predictions.jpg")
    print("   │   └── ...more predicted images")
    print("   └── labels/ (optional)")
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    run_inference()

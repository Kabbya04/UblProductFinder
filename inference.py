"""
Inference script for UBL vs Non-UBL YOLOv8 model.
This script allows you to run inference on individual images, directories of images,
or video files using your trained YOLOv8 model.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import time

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="UBL vs Non-UBL Detector Inference Script")
    parser.add_argument('--model', type=str, default="runs/detect/train2/weights/best.pt",
                        help='Path to the trained model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--output', type=str, default="output",
                        help='Path to output directory')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='0.5')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to *.txt file')
    parser.add_argument('--save-img', action='store_true', default=True,
                        help='Save annotated images')
    parser.add_argument('--show', action='store_true',
                        help='Display results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda device, i.e. 0 or cpu')
    
    return parser.parse_args()

def process_results(results, source_path, output_dir, save_img=True, save_txt=False, show=False):
    """Process and save inference results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, result in enumerate(results):
        # Get source filename
        if isinstance(source_path, (str, Path)):
            filename = os.path.basename(source_path)
            base_filename = os.path.splitext(filename)[0]
        else:
            # For video frame or webcam
            base_filename = f"frame_{i:06d}"
        
        # Get the original image
        img = result.orig_img
        
        # Get annotated image (with boxes)
        annotated_img = result.plot()
        
        # Save annotated image
        if save_img:
            output_path = os.path.join(output_dir, f"{base_filename}_annotated.jpg")
            cv2.imwrite(output_path, annotated_img)
            print(f"Saved annotated image to {output_path}")
        
        # Save detection results as text
        if save_txt:
            txt_path = os.path.join(output_dir, f"{base_filename}.txt")
            with open(txt_path, 'w') as f:
                for box in result.boxes:
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    # Convert xyxy to normalized xywh (YOLO format)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w, h = img.shape[1], img.shape[0]
                    
                    # YOLO format: class x_center y_center width height (normalized)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    f.write(f"{cls} {x_center} {y_center} {width} {height} {conf}\n")
            print(f"Saved detection results to {txt_path}")
        
        # Show the results
        if show:
            plt.figure(figsize=(10, 6))
            plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"UBL vs Non-UBL Detection - {base_filename}")
            plt.show()
    
    return len(results)

def run_inference(args):
    """Run inference with the trained model"""
    # Print banner
    print("\n" + "="*50)
    print("UBL vs Non-UBL Detector Inference")
    print("="*50)
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    # Print model information
    print(f"Model loaded: {model.names}")
    
    # Run inference
    print(f"Running inference on {args.source}...")
    start_time = time.time()
    
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=False,  # We'll handle saving ourselves
        device=args.device
    )
    
    # Process results
    num_processed = process_results(
        results, 
        args.source, 
        args.output, 
        save_img=args.save_img,
        save_txt=args.save_txt,
        show=args.show
    )
    
    # Print summary
    elapsed_time = time.time() - start_time
    print("\nInference Summary:")
    print(f"- Processed {num_processed} images/frames")
    print(f"- Time taken: {elapsed_time:.2f} seconds")
    print(f"- Output saved to: {os.path.abspath(args.output)}")
    print("="*50)

def main():
    """Main function"""
    args = parse_arguments()
    run_inference(args)

if __name__ == "__main__":
    main()
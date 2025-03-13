import cv2
import numpy as np
import time
from collections import deque
import torch

class BirdDetectionSystem:
    def __init__(self, 
                 model_path='yolov4_bird_detection.pt',
                 confidence_threshold=0.5,
                 change_detection_threshold=30,
                 history_frames=5,
                 tile_size=(416, 416),
                 overlap=0.2):
        """
        Initialize the bird detection system
        
        Parameters:
        model_path: Path to the trained YOLOv4 model
        confidence_threshold: Minimum confidence for bird detection
        change_detection_threshold: Pixel difference threshold for change detection
        history_frames: Number of frames to maintain for temporal analysis
        tile_size: Size of image tiles for processing
        overlap: Overlap percentage between tiles
        """
        self.confidence_threshold = confidence_threshold
        self.change_detection_threshold = change_detection_threshold
        
        # Initialize deque for frame history
        self.frame_history = deque(maxlen=history_frames)
        
        # Load YOLOv4 model
        self.model = self._load_model(model_path)
        
        # Tiling parameters
        self.tile_size = tile_size
        self.overlap = overlap
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False)
        
        # Initialize repellent system
        self.repellent_system = RepellentSystem()
        
        print("Bird Detection System initialized.")
    
    def _load_model(self, model_path):
        """Load the YOLOv4 model"""
        try:
            # Load YOLOv4 model using PyTorch
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
            model.conf = self.confidence_threshold
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback detection method")
            return None
    
    def _create_tiles(self, frame):
        """
        Split the frame into overlapping tiles
        Returns: List of (tile, (x, y, w, h)) where (x,y,w,h) is the tile position
        """
        height, width = frame.shape[:2]
        tile_h, tile_w = self.tile_size
        stride_h = int(tile_h * (1 - self.overlap))
        stride_w = int(tile_w * (1 - self.overlap))
        
        tiles = []
        
        for y in range(0, height - tile_h + 1, stride_h):
            for x in range(0, width - tile_w + 1, stride_w):
                tile = frame[y:y + tile_h, x:x + tile_w]
                tiles.append((tile, (x, y, tile_w, tile_h)))
        
        return tiles
    
    def detect_changes(prev_frame, curr_frame, threshold=30, min_area=100):
        """
        Detect changes between two consecutive frames
        
        Parameters:
            prev_frame: Previous video frame
            curr_frame: Current video frame
            threshold: Pixel difference threshold (0-255)
            min_area: Minimum contour area to consider
            
        Returns:
            change_mask: Binary mask of changed regions
            regions: List of (x, y, w, h) tuples for ROIs
        """
        # Ensure frames are in grayscale for efficient processing
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            curr_gray = curr_frame
        
        # Apply Gaussian blur to reduce noise (optional but recommended)
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        
        # Compute absolute difference between frames
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Apply threshold to identify significant changes
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise and fill holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Generate ROIs from contours
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))
        
        return mask, regions
    

    def detect_changes_frame_diff(self, prev_frame, curr_frame, threshold=None):
        """
        Detect changes between two consecutive frames using frame differencing
        
        Parameters:
            prev_frame: Previous video frame
            curr_frame: Current video frame
            threshold: Optional threshold override
            
        Returns:
            change_mask: Binary mask of changed regions
            regions: List of (x, y, w, h) tuples for ROIs
        """
        if threshold is None:
            threshold = self.change_detection_threshold
        
        # Ensure frames are in grayscale for efficient processing
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            curr_gray = curr_frame
        
        # Apply Gaussian blur to reduce noise
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        
        # Compute absolute difference between frames
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Apply threshold to identify significant changes
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise and fill holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Generate ROIs from contours
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))
        
        return mask, regions
    
    def detect_changes_bg_subtraction(self, frame, min_area=100):
        """
        Detect changes using background subtraction
        
        Parameters:
            frame: Current video frame
            min_area: Minimum contour area to consider
            
        Returns:
            change_mask: Binary mask of changed regions
            regions: List of (x, y, w, h) tuples for ROIs
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply threshold to get cleaner results
        _, thresh = cv2.threshold(fg_mask, self.change_detection_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Generate ROIs from contours
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Add padding to ROIs for better detection
                x_pad, y_pad = max(0, x-10), max(0, y-10)
                w_pad, h_pad = w+20, h+20
                regions.append((x_pad, y_pad, w_pad, h_pad))
        
        return mask, regions
        
    def _detect_birds_in_roi(self, frame, roi_mask):
        """
        Apply YOLOv4 detection only in regions of interest
        Returns: List of bounding boxes [(x, y, w, h, confidence)]
        """
        # If no model is loaded, use basic detection
        if self.model is None:
            return self._fallback_detection(frame, roi_mask)
        
        # Find contours in the ROI mask
        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            # Get bounding rectangle of contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract ROI and make prediction only if ROI is large enough
            if w > 20 and h > 20:  # Minimum size threshold
                roi = frame[y:y+h, x:x+w]
                
                # Make prediction with YOLOv4
                results = self.model(roi)
                
                # Process results
                if len(results.pred[0]) > 0:
                    for *box, conf, cls in results.pred[0].cpu().numpy():
                        # Adjust coordinates to original frame
                        x1, y1, x2, y2 = box
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Add to detections with adjusted coordinates
                        detections.append((
                            x + x1, y + y1, 
                            x2 - x1, y2 - y1, 
                            float(conf)
                        ))
        
        return detections
    
    def _fallback_detection(self, frame, roi_mask):
        """Fallback detection method when model is not available"""
        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:  # Filter by size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                
                # Filter by aspect ratio (birds are typically wider than tall)
                if 0.5 < aspect_ratio < 3:
                    # Use area as confidence score
                    confidence = min(area / 1000, 1.0)
                    detections.append((x, y, w, h, confidence))
        
        return detections
    
    def _apply_temporal_filtering(self, detections):
        """
        Filter detections using temporal information
        Returns: Filtered list of detections
        """
        # If we don't have enough history, return all detections
        if len(self.frame_history) < self.frame_history.maxlen:
            return detections
        
        filtered_detections = []
        
        for det in detections:
            x, y, w, h, conf = det
            
            # Count how many previous frames had detections in similar locations
            temporal_support = 0
            
            for prev_dets in self.frame_history:
                for prev_det in prev_dets:
                    px, py, pw, ph, _ = prev_det
                    
                    # Check if there's overlap between detections
                    if (x < px + pw and x + w > px and
                        y < py + ph and y + h > py):
                        temporal_support += 1
                        break
            
            # If detection appears in multiple frames, keep it
            if temporal_support >= 2 or conf > 0.8:
                # Adjust confidence based on temporal support
                adjusted_conf = min(conf * (1 + 0.1 * temporal_support), 1.0)
                filtered_detections.append((x, y, w, h, adjusted_conf))
        
        return filtered_detections
    
    def process_frame(self, frame, prev_frame=None):
        """
        Process a video frame to detect birds
        
        Parameters:
            frame: Current video frame
            prev_frame: Previous video frame (optional, for frame differencing)
            
        Returns: Frame with annotations, list of detections
        """
        # Make a copy of the frame for drawing
        result_frame = frame.copy()
        
        # Detect changes
        if prev_frame is not None:
            # Use frame differencing if previous frame is provided
            change_mask, regions = self.detect_changes_frame_diff(prev_frame, frame)
        else:
            # Use background subtraction otherwise
            change_mask, regions = self.detect_changes_bg_subtraction(frame)
        
        # Detect birds in regions with changes
        raw_detections = []
        for x, y, w, h in regions:
            # Ensure region is within frame bounds
            x, y = max(0, x), max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            # Extract region
            roi = frame[y:y+h, x:x+w]
            
            # Skip empty ROIs
            if roi.size == 0:
                continue
            
            # Process with YOLOv5 model
            if self.model is not None:
                # Save ROI as temporary file for YOLOv5
                roi_path = f"temp_roi_{x}_{y}.jpg"
                cv2.imwrite(roi_path, roi)
                
                # Run detection
                results = self.model(roi_path)
                
                # Process results
                for result in results:
                    if len(result.boxes) > 0:
                        for box in result.boxes:
                            # Get coordinates (relative to ROI)
                            roi_x1, roi_y1, roi_x2, roi_y2 = map(int, box.xyxy[0].tolist())
                            confidence = float(box.conf[0])
                            
                            # Convert to frame coordinates
                            x1, y1 = x + roi_x1, y + roi_y1
                            x2, y2 = x + roi_x2, y + roi_y2
                            
                            # Add to detections
                            raw_detections.append((x1, y1, x2-x1, y2-y1, confidence))
                
                # Cleanup temp file
                try:
                    import os
                    os.remove(roi_path)
                except:
                    pass
            else:
                # Use fallback detection if model not available
                fallback_dets = self._fallback_detection(roi, np.ones((h, w), dtype=np.uint8)*255)
                # Adjust coordinates to original frame
                for fx, fy, fw, fh, conf in fallback_dets:
                    raw_detections.append((x+fx, y+fy, fw, fh, conf))
        
        # Apply temporal filtering
        self.frame_history.append(raw_detections)
        filtered_detections = self._apply_temporal_filtering(raw_detections)
        
        # Draw detections on the result frame
        for x, y, w, h, conf in filtered_detections:
            if conf >= self.confidence_threshold:
                # Draw bounding box
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw confidence
                label = f"Bird: {conf:.2f}"
                cv2.putText(result_frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Activate repellent if birds detected
        if any(conf >= self.confidence_threshold for _, _, _, _, conf in filtered_detections):
            self.repellent_system.activate(
                num_birds=len(filtered_detections),
                locations=[(x + w/2, y + h/2) for x, y, w, h, _ in filtered_detections]
            )
        else:
            self.repellent_system.deactivate()
        
        return result_frame, filtered_detections
    
    def benchmark_performance(self, video_path, frames_to_process=100):
        """
        Benchmark performance with and without change detection
        
        Parameters:
            video_path: Path to input video
            frames_to_process: Number of frames to process
            
        Returns:
            Dictionary with performance statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        times_without_cd = []
        times_with_cd = []
        
        prev_frame = None
        frame_count = 0
        
        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            if prev_frame is None:
                prev_frame = frame.copy()
                continue
            
            # Measure time without change detection (direct YOLO)
            start_time = time.time()
            if self.model is not None:
                _ = self.model(frame)
            end_time = time.time()
            times_without_cd.append(end_time - start_time)
            
            # Measure time with change detection
            start_time = time.time()
            _, _ = self.process_frame(frame, prev_frame)
            end_time = time.time()
            times_with_cd.append(end_time - start_time)
            
            prev_frame = frame.copy()
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics
        avg_time_without_cd = sum(times_without_cd) / len(times_without_cd)
        avg_time_with_cd = sum(times_with_cd) / len(times_with_cd)
        speedup = avg_time_without_cd / avg_time_with_cd
        
        print(f"Average processing time without change detection: {avg_time_without_cd*1000:.2f} ms")
        print(f"Average processing time with change detection: {avg_time_with_cd*1000:.2f} ms")
        print(f"Speedup factor: {speedup:.2f}x")
        
        return {
            'without_cd': avg_time_without_cd,
            'with_cd': avg_time_with_cd,
            'speedup': speedup
        }
    
    def process_video(self, video_path, output_path=None):
        """Process a video file with frame differencing"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            if prev_frame is not None:
                result_frame, detections = self.process_frame(frame, prev_frame)
            else:
                result_frame, detections = self.process_frame(frame)
            
            # Write to output if writer is initialized
            if writer:
                writer.write(result_frame)
            
            # Store previous frame
            prev_frame = frame.copy()
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        
        print(f"Video processing complete. Processed {frame_count} frames.")



    import matplotlib.pyplot as plt
    import cv2

    def show_image(img):
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show() 
        
    def process_camera_feed(self, camera_id=0):
        """Process live camera feed"""
        cap = cv2.VideoCapture(camera_id)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            result_frame, detections = self.process_frame(frame)
            
            # Display the result
            show_image(result_frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

       


    def generate_detection_statistics(self, video_path, duration_seconds=60):
        """
        Process a portion of a video and generate detection statistics
        
        Parameters:
            video_path: Path to input video
            duration_seconds: How many seconds of video to process
            
        Returns:
            Dictionary with detection statistics
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames_to_process = int(fps * duration_seconds)
        
        prev_frame = None
        frame_count = 0
        total_detections = 0
        frames_with_birds = 0
        detection_counts = []
        processing_times = []
        
        while frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Process the frame
            if prev_frame is not None:
                _, detections = self.process_frame(frame, prev_frame)
            else:
                _, detections = self.process_frame(frame)
            
            # Record processing time
            processing_times.append(time.time() - start_time)
            
            # Record detection statistics
            detections_above_threshold = [d for d in detections if d[4] >= self.confidence_threshold]
            detection_counts.append(len(detections_above_threshold))
            
            if detections_above_threshold:
                frames_with_birds += 1
                total_detections += len(detections_above_threshold)
            
            # Store previous frame
            prev_frame = frame.copy()
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        bird_detection_rate = frames_with_birds / frame_count if frame_count > 0 else 0
        avg_birds_per_detection = total_detections / frames_with_birds if frames_with_birds > 0 else 0
        
        statistics = {
            'frames_processed': frame_count,
            'frames_with_birds': frames_with_birds,
            'total_detections': total_detections,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'bird_detection_rate': bird_detection_rate,
            'avg_birds_per_detection': avg_birds_per_detection,
            'max_birds_in_frame': max(detection_counts) if detection_counts else 0
        }
        
        print("Detection Statistics:")
        for key, value in statistics.items():
            print(f"  {key}: {value}")
        
        return statistics   


class RepellentSystem:
    """System to repel birds when detected"""
    
    def __init__(self, cooldown_period=10):
        """
        Initialize the repellent system
        
        Parameters:
        cooldown_period: Minimum time between activations (seconds)
        """
        self.cooldown_period = cooldown_period
        self.last_activation_time = 0
        self.active = False
        
        # Available repellent methods
        self.repellent_methods = {
            'sound': self._activate_sound,
            'light': self._activate_light,
            'mechanical': self._activate_mechanical
        }
    
    def activate(self, num_birds, locations):
        """
        Activate appropriate repellent based on detection
        
        Parameters:
        num_birds: Number of birds detected
        locations: List of (x, y) center points of bird detections
        """
        current_time = time.time()
        
        # Check if cooldown period has passed
        if not self.active and current_time - self.last_activation_time > self.cooldown_period:
            self.active = True
            self.last_activation_time = current_time
            
            # Choose repellent method based on number of birds
            if num_birds <= 2:
                self.repellent_methods['sound'](locations)
            elif num_birds <= 5:
                self.repellent_methods['light'](locations)
            else:
                self.repellent_methods['mechanical'](locations)
            
            print(f"Repellent activated: {num_birds} birds detected")
    
    def deactivate(self):
        """Deactivate all repellent methods"""
        if self.active:
            # Deactivate all methods
            self._deactivate_sound()
            self._deactivate_light()
            self._deactivate_mechanical()
            
            self.active = False
    
    def _activate_sound(self, locations):
        """Activate sound-based repellent"""
        # Code to activate speakers, play distress calls, etc.
        print("Sound repellent activated")
    
    def _deactivate_sound(self):
        """Deactivate sound-based repellent"""
        # Code to stop sounds
        pass
    
    def _activate_light(self, locations):
        """Activate light-based repellent"""
        # Code to activate strobe lights, lasers, etc.
        print("Light repellent activated")
    
    def _deactivate_light(self):
        """Deactivate light-based repellent"""
        # Code to turn off lights
        pass
    
    def _activate_mechanical(self, locations):
        """Activate mechanical repellent"""
        # Code to activate physical deterrents, drones, etc.
        print("Mechanical repellent activated")
    
    def _deactivate_mechanical(self):
        """Deactivate mechanical repellent"""
        # Code to deactivate mechanical systems
        pass


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bird Detection System")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--stats", action="store_true", help="Generate detection statistics")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds to process")
    parser.add_argument("--model", type=str, default="models/yolov4_bird_detection.pt", 
                        help="Path to YOLOv5 model")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize the detection system
    detector = BirdDetectionSystem(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    if args.video and args.stats:
        detector.generate_detection_statistics(args.video, args.duration)
    else:
        print("Please provide a video file path with --video and use --stats")
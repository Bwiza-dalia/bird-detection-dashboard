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
    
    def _detect_changes(self, frame):
        """
        Detect changes in the current frame compared to history
        Returns: A binary mask of changed regions
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Threshold to get binary mask
        _, change_mask = cv2.threshold(
            fg_mask, self.change_detection_threshold, 255, cv2.THRESH_BINARY)
        
        return change_mask
    
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
    
    def process_frame(self, frame):
        """
        Process a video frame to detect birds
        Returns: Frame with annotations, list of detections
        """
        # Make a copy of the frame for drawing
        result_frame = frame.copy()
        
        # Detect changes compared to background
        change_mask = self._detect_changes(frame)
        
        # Detect birds in regions with changes
        raw_detections = self._detect_birds_in_roi(frame, change_mask)
        
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
    
    def process_video(self, video_path, output_path=None):
        """Process a video file"""
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
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            result_frame, detections = self.process_frame(frame)
            
            # Write to output if writer is initialized
            if writer:
                writer.write(result_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")
        
        # Release resources
        cap.release()
        if writer:
            writer.write(result_frame)
        
        print(f"Video processing complete. Processed {frame_count} frames.")
        
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
            cv2.imshow('Bird Detection', result_frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


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
    # Initialize the detection system
    detector = BirdDetectionSystem(
        model_path='path/to/yolov4_bird_model.pt',
        confidence_threshold=0.6
    )
    
    # Process video file
    # detector.process_video('path/to/video.mp4', 'output.avi')
    
    # Or process camera feed
    detector.process_camera_feed()
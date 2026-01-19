import os
import json
import faiss
import numpy as np
import cv2

class FaceDatabase:
    def __init__(self, index_path="faces.index", json_path="id_to_name.json"):
        self.index_path = index_path
        self.json_path = json_path
        
        # Load metadata first to assist with potential migration
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.id_to_name = json.load(f)
        else:
            self.id_to_name = {}
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            
            # Capability Check: Does this index support remove_ids?
            needs_migration = False
            try:
                # Try to remove a non-existent ID to test capability
                # ID -1 is usually safe to try removing
                self.index.remove_ids(np.array([-1], dtype=np.int64))
            except RuntimeError:
                print("Index does not support remove_ids (likely HNSW). Scheduling migration...")
                needs_migration = True
            except Exception as e:
                print(f"Unexpected error checking index capability: {e}")
            
            # Also migrate if it's the old IndexIDMap (Type 1)
            if isinstance(self.index, faiss.IndexIDMap) and not isinstance(self.index, faiss.IndexIDMap2):
                 needs_migration = True

            if needs_migration:
                print("Migrating database to IndexIDMap2 with FlatIP...")
                try:
                    # Create new IndexIDMap2 with FlatIP (same as original design)
                    new_inner = faiss.IndexFlatIP(512)
                    new_index = faiss.IndexIDMap2(new_inner)
                    
                    # Reconstruct and transfer all faces
                    ids = [int(k) for k in self.id_to_name.keys() if k.isdigit()]
                    count = 0
                    for pid in ids:
                        try:
                            vec = self.index.reconstruct(pid)
                            new_index.add_with_ids(
                                vec.reshape(1, -1).astype(np.float32),
                                np.array([pid], dtype=np.int64)
                            )
                            count += 1
                        except Exception as e:
                            print(f"Skipping ID {pid}: {e}")
                            
                    print(f"Migration complete. Transferred {count} faces.")
                    self.index = new_index
                    faiss.write_index(self.index, self.index_path)
                except Exception as e:
                    print(f"Migration failed: {e}")
                    
        else:
            # Initialize empty index (512D for ArcFace)
            # We wrap it in IDMap2 to support remove_ids
            index = faiss.IndexFlatIP(512)
            self.index = faiss.IndexIDMap2(index)
            
        self.next_id = 0
        if self.id_to_name:
            ids = [int(k) for k in self.id_to_name.keys() if k.isdigit()]
            if ids:
                self.next_id = max(ids) + 1

    def add_face(self, embedding, name):
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        # Prepare data
        emb_vector = embedding.reshape(1, -1).astype(np.float32)
        id_vector = np.array([self.next_id], dtype=np.int64)
        
        # Add to FAISS
        # Check if index requires IDs (IndexIDMap) or generates them (IndexFlat)
        # The safest way is to check for the add_with_ids method
        if hasattr(self.index, 'add_with_ids'):
            self.index.add_with_ids(emb_vector, id_vector)
        else:
            self.index.add(emb_vector)
        
        # Add to ID mapping
        self.id_to_name[str(self.next_id)] = name
        self.next_id += 1
        return self.next_id - 1

    def prune_embeddings(self, name, max_faces=100):
        # Find all IDs for this person
        person_ids = [int(id_str) for id_str, n in self.id_to_name.items() if n == name]
        
        if len(person_ids) <= max_faces:
            return

        print(f"Pruning faces for {name}. Current: {len(person_ids)}, Max: {max_faces}")
        
        # Retrieve embeddings
        embeddings = []
        valid_ids = []
        
        for pid in person_ids:
            try:
                # reconstruct returns the vector for the given ID
                vec = self.index.reconstruct(pid)
                embeddings.append(vec)
                valid_ids.append(pid)
            except RuntimeError:
                # ID might not exist in index (sync issue?)
                continue
                
        if not embeddings:
            return

        embeddings = np.array(embeddings)
        
        # Calculate mean center
        center = np.mean(embeddings, axis=0)
        # Normalize center
        norm = np.linalg.norm(center)
        if norm > 0:
            center = center / norm
            
        # Calculate similarities to center (Inner Product)
        # embeddings are already normalized (assumed from add_face)
        similarities = np.dot(embeddings, center)
        
        # Find indices to remove (lowest similarity = farthest)
        num_to_remove = len(valid_ids) - max_faces
        if num_to_remove > 0:
            # argsort returns indices that would sort the array
            # The first num_to_remove elements are the smallest similarities
            sorted_indices = np.argsort(similarities)
            remove_indices = sorted_indices[:num_to_remove]
            
            ids_to_remove = [valid_ids[i] for i in remove_indices]
            
            # Remove from FAISS
            self.index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            
            # Remove from metadata
            for pid in ids_to_remove:
                del self.id_to_name[str(pid)]
                
            print(f"Removed {len(ids_to_remove)} outlier faces for {name}")

    def save(self):
        # Create directory if it doesn't exist
        index_dir = os.path.dirname(self.index_path)
        if index_dir and not os.path.exists(index_dir):
            os.makedirs(index_dir, exist_ok=True)
        json_dir = os.path.dirname(self.json_path)
        if json_dir and not os.path.exists(json_dir):
            os.makedirs(json_dir, exist_ok=True)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.json_path, 'w') as f:
            json.dump(self.id_to_name, f, indent=4)
        print(f"Database saved with {self.index.ntotal} faces")



def adaptive_threshold_knn(distances, indices, k=5):
    """
    Calculate adaptive threshold based on k-NN distances
    Returns: threshold, is_ambiguous
    """
    # Filter out invalid results (FAISS returns -1 for empty slots)
    valid_mask = indices != -1
    valid_distances = distances[valid_mask]
    
    if len(valid_distances) == 0:
        return 1.0, False, {}

    # Convert distances to similarities
    similarities = (valid_distances + 1) / 2
    
    # Calculate statistics
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities) if len(similarities) > 1 else 0.0
    max_sim = np.max(similarities)
    min_sim = np.min(similarities)
    
    # Global Floor for Unknown Rejection (lowered from 0.7 to 0.55 for tilted faces)
    if max_sim < 0.55:
        return 1.0, False, {} # Reject all
    
    # Rule 1: Clear Winner (High confidence)
    # If we have very few faces in DB, mean might equal max, so we relax the spread check
    is_clear_winner = False
    if len(similarities) == 1:
        if max_sim > 0.70: is_clear_winner = True
    elif max_sim > 0.70 and (max_sim - mean_sim) > 0.12:
        is_clear_winner = True

    if is_clear_winner:
        threshold = max_sim - 0.08
        ambiguous = False
    
    # Rule 2: Consistent Cluster (Moderate confidence)
    elif max_sim > 0.65 and std_sim < 0.1:
        threshold = max_sim - 0.08
        ambiguous = False
    
    # Rule 3: Ambiguous / Potential Confusion
    elif max_sim > 0.55:
        threshold = max_sim # Only accept the very best match
        ambiguous = True
    
    # Rule 4: Weak Match
    else:
        threshold = 0.75 # High threshold to reject
        ambiguous = True
    
    return threshold, ambiguous, {
        'mean_similarity': mean_sim,
        'std_similarity': std_sim,
        'max_similarity': max_sim,
        'spread': max_sim - min_sim
    }

def smart_knn_recognition(index, query_embedding, id_to_name, k=5):
    """Smart recognition with adaptive thresholding"""
    # Normalize query embedding before search
    query = query_embedding.reshape(1, -1).astype(np.float32)
    norm = np.linalg.norm(query)
    if norm > 0:
        query = query / norm
    
    # Search for k neighbors
    distances, indices = index.search(query, k)
    
    # Calculate adaptive threshold
    # Pass indices to filter out garbage
    threshold, is_ambiguous, stats = adaptive_threshold_knn(distances[0], indices[0], k)
    
    # Filter matches above threshold
    matches = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            similarity = (dist + 1) / 2
            if similarity >= threshold:
                name = id_to_name.get(str(idx), "unknown")
                matches.append({
                    'id': int(idx),
                    'name': name,
                    'similarity': float(similarity)
                })
    
    # Determine result
    if not matches:
        result_name = "unknown"
        confidence = 0.0
    elif is_ambiguous:
        # Use weighted average of top matches
        top_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:3]
        
        # Check if top matches agree on the name
        names = [m['name'] for m in top_matches]
        if len(set(names)) == 1:
            # All agree
            result_name = names[0]
            confidence = sum(m['similarity'] for m in top_matches) / len(top_matches)
        else:
            # Disagreement
            best_match = top_matches[0]
            # Lowered strictness slightly since we fixed alignment
            if best_match['similarity'] > 0.70:
                result_name = best_match['name']
                confidence = best_match['similarity'] * 0.9 # Less penalty
            else:
                result_name = "unknown"
                confidence = 0.0
    else:
        # Clear best match
        result_name = matches[0]['name']
        confidence = matches[0]['similarity']
        
    # Final safety check for very low confidence (lowered from 0.60 to 0.50)
    if confidence < 0.50:
        result_name = "unknown"
    
    return {
        'name': result_name,
        'confidence': confidence,
        'is_ambiguous': is_ambiguous,
        'threshold_used': threshold,
        'matches_found': len(matches),
        'top_matches': matches[:3],
        'stats': stats
    }

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Determine intersection rectangle
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    if intersection_area <= 0:
        return 0.0
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def main():
    # Load FAISS index and metadata
    print("Loading Face Database...")
    face_db = FaceDatabase()
    index = face_db.index
    id_to_name = face_db.id_to_name
    print(f"Loaded database with {index.ntotal} faces")

    # FPS tracking
    fps_start_time = time.time()
    fps_counter = 0
    actual_fps = 0
    frame_times = []

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Camera settings for real-time processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

    # Warm up camera
    print("Warming up camera...")
    for _ in range(10):
        ret, _ = cap.read()

    print("Starting real-time face recognition. Press 'q' to quit...")
    print("Press 't' to toggle threshold, 's' to save frame")

    # Recognition settings
    SIMILARITY_THRESHOLD = 0.5  # Adjust this value for sensitivity
    use_simple_detection = False  # Toggle between simple and improved detection

    # Frame skipping settings
    frame_count = 0
    SKIP_FRAMES = 10
    tracked_faces = []  # Stores {'bbox': (x1,y1,x2,y2), 'name': str, 'similarity': float, 'color': tuple}
    learning_tracker = {} # {name: count}

    while True:
        frame_start = time.time()
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        
        # OPTIMIZATION: Convert to RGB ONCE here
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Always detect faces first
        current_detections = []
        
        if use_simple_detection:
            # Simple detection (Pass RGB)
            detection = face_detector.detect(frame_rgb)
            if detection.detections:
                for face in detection.detections:
                    bbox = face.location_data.relative_bounding_box
                    start_point = (int(bbox.xmin * frame.shape[1]), int(bbox.ymin * frame.shape[0]))
                    end_point = (int((bbox.xmin + bbox.width) * frame.shape[1]), 
                            int((bbox.ymin + bbox.height) * frame.shape[0]))
                    current_detections.append((start_point[0], start_point[1], end_point[0], end_point[1]))
        else:
            # Improved detection (Pass RGB)
            faces = face_detector.detect_improved(frame_rgb)
            for face in faces:
                current_detections.append(face['bbox'])
                
        # 2. Process recognition or tracking
        new_tracked_faces = []
        
        if frame_count % SKIP_FRAMES == 0:
            # Run full recognition
            for bbox in current_detections:
                x1, y1, x2, y2 = bbox
                
                # Extract face crop from RGB frame
                face_crop_rgb = frame_rgb[y1:y2, x1:x2]
                
                if face_crop_rgb.size > 0 and face_crop_rgb.shape[0] > 20 and face_crop_rgb.shape[1] > 20:
                    # Pass RGB crop to embedder
                    emb = face_embedder.embed(face_crop_rgb)
                    
                    name = "unknown"
                    similarity = 0.0
                    color = (0, 0, 255)
                    
                    if emb is not None:
                        result = smart_knn_recognition(index, emb, id_to_name)
                        name = result['name']
                        similarity = result['confidence']
                        
                        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                        if result['is_ambiguous'] and name != "unknown":
                            color = (0, 255, 255)
                        
                        # Adaptive Learning Logic
                        # Disabled: Only add unknown faces manually (via 'n' key)
                        if name != "unknown" and similarity > 0.75:
                            learning_tracker[name] = learning_tracker.get(name, 0) + 1
                            if learning_tracker[name] > 5: # 5 consecutive checks (approx 25 frames)
                                # Redundancy Check: Only add if the new face provides NEW information.
                                # If similarity is too high (> 0.8), we already have a near-identical template.
                                if similarity < 0.8:
                                    print(f"Updating template for {name} (sim: {similarity:.2f})...")
                                    # Use the class method which now handles IDs correctly
                                    face_db.add_face(emb, name)
                                    face_db.prune_embeddings(name, max_faces=100)
                                    face_db.save()
                                learning_tracker[name] = 0 # Reset
                        else:
                            if name in learning_tracker:
                                learning_tracker[name] = 0
                    
                    new_tracked_faces.append({
                        'bbox': bbox,
                        'name': name,
                        'similarity': similarity,
                        'color': color,
                        'embedding': emb
                    })
        else:
            # Tracking mode: Match current detections to previous tracked faces
            for bbox in current_detections:
                best_iou = 0
                best_match = None
                
                # Find best matching face from previous frame
                for tracked in tracked_faces:
                    iou = calculate_iou(bbox, tracked['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = tracked
                
                # If match found (IoU > 0.3), inherit identity
                if best_iou > 0.3 and best_match:
                    new_tracked_faces.append({
                        'bbox': bbox,  # Update position
                        'name': best_match['name'],
                        'similarity': best_match['similarity'],
                        'color': best_match['color'],
                        'embedding': best_match.get('embedding')
                    })
                else:
                    # New face detected during skip frames - mark as unknown or wait for next recognition
                    new_tracked_faces.append({
                        'bbox': bbox,
                        'name': "scanning...",
                        'similarity': 0.0,
                        'color': (200, 200, 200),
                        'embedding': None
                    })
        
        # Update tracked faces
        tracked_faces = new_tracked_faces
        
        # 3. Draw results
        for face in tracked_faces:
            x1, y1, x2, y2 = face['bbox']
            color = face['color']
            name = face['name']
            similarity = face['similarity']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if name == "scanning...":
                label = name
            else:
                label = f'{name} ({similarity:.2f})'
                
            cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate FPS
        frame_count += 1
        fps_counter += 1
        current_time = time.time()
        elapsed_time = current_time - fps_start_time
        
        # Calculate frame processing time
        frame_time = (current_time - frame_start) * 1000  # Convert to ms
        frame_times.append(frame_time)
        if len(frame_times) > 30:  # Keep last 30 frames
            frame_times.pop(0)
        
        # Update FPS every 0.5 seconds
        if elapsed_time > 0.5:
            actual_fps = fps_counter / elapsed_time
            fps_counter = 0
            fps_start_time = current_time
        
        # Display performance info
        avg_frame_time = np.mean(frame_times) if frame_times else 0
        
        # Top-left info
        cv2.putText(frame, f'FPS: {actual_fps:.1f}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Frame: {frame_time:.1f}ms', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f'Avg: {avg_frame_time:.1f}ms', (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Top-right info
        info_y = 30
        cv2.putText(frame, f'Threshold: {SIMILARITY_THRESHOLD}', 
                    (frame.shape[1] - 250, info_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Mode: {"Enhanced" if not use_simple_detection else "Simple"}', 
                    (frame.shape[1] - 250, info_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Faces: {index.ntotal}', 
                    (frame.shape[1] - 250, info_y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Bottom-left controls hint
        cv2.putText(frame, "Controls: q=quit, t=thresh, s=save, d=detect, n=new friend", 
                    (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Show frame
        cv2.imshow('Face Recognition', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('t'):
            # Toggle threshold
            if SIMILARITY_THRESHOLD == 0.5:
                SIMILARITY_THRESHOLD = 0.6
            elif SIMILARITY_THRESHOLD == 0.6:
                SIMILARITY_THRESHOLD = 0.4
            else:
                SIMILARITY_THRESHOLD = 0.5
            print(f"Threshold changed to: {SIMILARITY_THRESHOLD}")
        elif key == ord('s'):
            # Save current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Frame saved as {filename}")
        elif key == ord('n'):
            # Dynamic Enrollment
            print("Dynamic Enrollment Mode")
            if tracked_faces:
                # Iterate through all detected faces
                for i, face in enumerate(tracked_faces):
                    if face.get('embedding') is None:
                        continue
                    
                    # Highlight the specific face being queried
                    x1, y1, x2, y2 = face['bbox']
                    display_frame = frame.copy()
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(display_frame, f"Face {i+1}/{len(tracked_faces)}: {face['name']}", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    cv2.imshow('Face Recognition', display_frame)
                    cv2.waitKey(1)
                    
                    # Ask user for action
                    print(f"\n[Face {i+1}] Identified as: {face['name']}")
                    choice = input("Add this face to database? (y/n/q): ").lower().strip()
                    
                    if choice == 'q':
                        break
                    elif choice == 'y':
                        try:
                            default_name = face['name'] if face['name'] != "unknown" else ""
                            prompt = f"Enter name (default: {default_name}): " if default_name else "Enter name: "
                            new_name = input(prompt).strip()
                            
                            final_name = new_name if new_name else default_name
                            
                            if not final_name or final_name == "unknown":
                                print("Invalid name. Skipping.")
                                continue
                                
                            face_db.add_face(face['embedding'], final_name)
                            face_db.prune_embeddings(final_name, max_faces=100)
                            face_db.save()
                            print(f"Added {final_name} to database.")
                            face['name'] = final_name
                            face['color'] = (0, 255, 0)
                        except Exception as e:
                            print(f"Error enrolling: {e}")
                    else:
                        print("Skipped.")
            else:
                print("No faces detected to enroll.")
        elif key == ord('d'):
            # Toggle detection mode
            use_simple_detection = not use_simple_detection
            mode = "Simple" if use_simple_detection else "Enhanced"
            print(f"Detection mode changed to: {mode}")
        elif key == ord('+') or key == ord('='):
            # Increase threshold
            SIMILARITY_THRESHOLD = min(0.9, SIMILARITY_THRESHOLD + 0.05)
            print(f"Threshold increased to: {SIMILARITY_THRESHOLD:.2f}")
        elif key == ord('-') or key == ord('_'):
            # Decrease threshold
            SIMILARITY_THRESHOLD = max(0.1, SIMILARITY_THRESHOLD - 0.05)
            print(f"Threshold decreased to: {SIMILARITY_THRESHOLD:.2f}")

    cap.release()
    cv2.destroyAllWindows()

    print("Face recognition stopped")

if __name__ == "__main__":
    main()
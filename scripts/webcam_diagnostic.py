import asyncio
import sys
import time
from uuid import uuid4
from ml.SessionRunner import SessionRunner

async def main():
    print("Starting webcam diagnostic...")
    session_id = uuid4()
    
    # Track statistics
    frames_processed = 0
    face_detected_count = 0
    crop_dims = []
    nonzero_au_counts = []
    emotions = []
    
    async with SessionRunner(session_id=session_id, fps=15) as runner:
        print("Warmup (2 seconds)...")
        await asyncio.sleep(2.0)
        print("Capturing 60 frames...")
        
        for _ in range(60):
            t0 = time.time()
            await runner._tick()
            
            # Extract info
            face_pipe = runner._face
            pred = runner.latest_prediction
            
            frames_processed += 1
            if face_pipe.last_face_bbox is not None:
                face_detected_count += 1
                # calculate crop dims - we can't easily get the crop here since it's local in _tick
                # but we can see AUs
            
            # Check AUs directly if possible, but they are written to DB.
            # We can monkeypatch or just check if prediction source is "pretrained"
            print(f"Frame {frames_processed}:")
            print(f"  Emotion: {pred.emotion} (src={pred.emotion_source}) probs: {pred.emotion_scores}")
            print(f"  Behaviors: stress={pred.stress:.3f}, eng={pred.engagement:.3f}, att={pred.attention:.3f}, fat={pred.fatigue:.3f}")
            emotions.append(pred.emotion)
            
            elapsed = time.time() - t0
            sleep_time = max(0.0, 1.0/15.0 - elapsed)
            await asyncio.sleep(sleep_time)

    print("\n--- Summary ---")
    print(f"Total Frames Processed: {frames_processed}")
    print(f"Face Detected Rate: {face_detected_count}/{frames_processed} ({(face_detected_count/frames_processed)*100:.1f}%)")
    print(f"Emotions seen: {set(emotions)}")
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())

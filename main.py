import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
import cv2
import io
import os
import cloudinary
from cloudinary import CloudinaryImage, uploader
from dotenv import load_dotenv


load_dotenv()


app = FastAPI()

# Load both models
acne_model = YOLO('acne_model.pt')
eyecircle_model = YOLO('eyecircle_model.pt')

# Configure Cloudinary

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)



# file to link
def file_to_link(file):
    return uploader.upload(file, resource_type="auto")['url']


@app.post("/predict/combined/img")
async def predict_combined(file: UploadFile = File(...)):
    try:
        # Verify file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Run both models
        acne_results = acne_model.predict(image)
        eyecircle_results = eyecircle_model.predict(image)

        # Draw bounding boxes
        for box in acne_results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"Acne: {conf:.2f}", (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for box in eyecircle_results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(image, f"Eyecircle: {conf:.2f}", (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Convert image to base64
        _, buffer = cv2.imencode('.jpg', image)
        # img_str = base64.b64encode(buffer).decode()
        img_byte = io.BytesIO(buffer)
        img_str = file_to_link(img_byte)

        return {
            "status": "success",
            "acne_detections": acne_results[0].boxes.data.tolist(),
            "eyecircle_detections": eyecircle_results[0].boxes.data.tolist(),
            "annotated_image": img_str
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/combined/")
async def predict_combined(file: UploadFile = File(...)):
    # Read and process the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run both models
    acne_results = acne_model.predict(image)
    eyecircle_results = eyecircle_model.predict(image)
    
    return {
        "acne_detections": acne_results[0].boxes.data.tolist(),
        "eyecircle_detections": eyecircle_results[0].boxes.data.tolist()
    }

async def process_video_frame(frame):
    # Run both models on the frame
    acne_results = acne_model.predict(frame)
    eyecircle_results = eyecircle_model.predict(frame)

    # Draw bounding boxes for acne detections
    for box in acne_results[0].boxes.data.tolist():  # Accessing the first item in the list
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Acne: {conf:.2f}", (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw bounding boxes for eyecircle detections
    for box in eyecircle_results[0].boxes.data.tolist():  # Accessing the first item in the list
        x1, y1, x2, y2, conf, cls = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"Eyecircle: {conf:.2f}", (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame


@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    try:
        # Verify file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Create input and output file paths
        input_filename = f"input_{file.filename}"
        output_filename = f"processed_{file.filename}"
        
        # Save uploaded file
        with open(input_filename, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process video
        cap = cv2.VideoCapture(input_filename)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = await process_video_frame(frame)
            out.write(processed_frame)
        
        cap.release()
        out.release()
        
        # Upload processed video to Cloudinary
        with open(output_filename, "rb") as video_file:
            video_url = file_to_link(video_file)

        return {
            "status": "success",
            "video_url": video_url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up input file
        if os.path.exists(input_filename):
            os.remove(input_filename)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Verify file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Process with models
        acne_results = acne_model.predict(image)
        eyecircle_results = eyecircle_model.predict(image)
        
        return {
            "status": "success",
            "acne_detections": acne_results[0].boxes.data.tolist(),
            "eyecircle_detections": eyecircle_results[0].boxes.data.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Test endpoint
@app.get("/")
async def root():
    return {"message": "API is running"}


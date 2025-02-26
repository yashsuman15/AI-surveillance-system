from transformers import Owlv2Processor, Owlv2ForObjectDetection
import cv2
import torch
from PIL import Image
import numpy as np
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
import matplotlib.pyplot as plt

# Initialize the processor and model
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble", cache_dir="/path/to/local/cache")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", cache_dir="/path/to/local/cache")

# Function to preprocess image
def get_preprocessed_image(pixel_values):
    pixel_values = pixel_values.squeeze().numpy()
    unnormalized_image = (pixel_values * np.array(OPENAI_CLIP_STD)[:, None, None]) + np.array(OPENAI_CLIP_MEAN)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
    unnormalized_image = Image.fromarray(unnormalized_image)
    return unnormalized_image

# Function to process video frames
def process_video(video_path, texts):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Process the image
        inputs = processor(text=texts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get unnormalized image
        unnormalized_image = get_preprocessed_image(inputs.pixel_values)
        
        # Convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
        results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)
        
        # Retrieve predictions for the first image for the corresponding text queries
        i = 0
        text = texts[i]
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        
        # Draw bounding boxes on the frame
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = tuple(box)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            cv2.putText(frame, f"{text[label]}: {round(score.item(), 3)}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# Function to process a single frame and return the frame with bounding boxes
def OWLv2_process_frame(frame, texts):
    # Convert frame to PIL image
    image = Image.fromarray(frame)
    
    # Process the image
    inputs = processor(text=texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.2)
    
    # Retrieve predictions for the first image for the corresponding text queries
    i = 0
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    
    # Draw bounding boxes on the frame
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"{texts[label]}: {score:.3f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame


if __name__ == "__main__":
    # Define the texts to detect
    texts = [['man', 'human', 'child', 'animal']]
    image = r"sample2.jpg"
    frame = cv2.imread(image)
    frame = OWLv2_process_frame(frame, texts[0])
    
    plt.imshow(frame)
    plt.show()
    
    # Path to the video file
    # video_path = 'sample-media.mp4'
    # Process the video
    # process_video(video_path, texts)
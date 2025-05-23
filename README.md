 3Dify
------

Photo to 3D Model Converter 🖼️➡️📦
This project converts a 2D photo (with a single object like a chair, toy, or car) into a simple 3D model (.obj or .stl), with automatic background removal and depth map estimation using AI.


![Screenshot 2025-05-03 153445](https://github.com/user-attachments/assets/ca07df47-f4e6-49ad-bedf-9324035d5c8e)

![Screenshot 2025-05-03 153454](https://github.com/user-attachments/assets/c987d553-9999-41b6-989c-0fc63509f133)

![Screenshot 2025-05-03 153505](https://github.com/user-attachments/assets/85363afe-39de-4431-955c-46b4d9a18c83)

![Screenshot 2025-05-03 153532](https://github.com/user-attachments/assets/2bb40edd-7a27-4908-85d4-382ad271e81c)


   🚀 Steps to Run
   
    1) Install dependencies
        Make sure you have Python installed, then install all required libraries with:
            ⇨ pip install -r requirements.txt

    2) Run the conversion
        Use the following command to convert an image to a 3D model:
            ⇨ python photo-to-3d.py path/to/your/image.jpg --format obj
            (Replace path/to/your/image.jpg with the actual path to your image file.)

    🎯 The default output format is .obj. You can also use --format stl to export as an STL model.

Output📦

    - The script will display progress in the terminal.

    - A background-removed image (extracted_object.png) will be saved in your workspace.

    - The generated 3D model will be saved as yourfilename_3d.obj or .stl.


📦 Libraries Used
    
    ⇛ numpy – Numerical operations
    
    ⇛ opencv-python – Saving images
    
    ⇛ torch – Backend for AI models
    
    ⇛ Pillow – Image processing
    
    ⇛ trimesh – 3D mesh creation and export
    
    ⇛ rembg – Background removal
    
    ⇛ transformers – Depth estimation pipeline
    
    ⇛ onnxruntime – Model inference backend

🧠 Thought Process (in short)

The idea was to build a simple yet functional pipeline to generate a rough 3D model from a single 2D image. First, the background is removed to isolate the object. Then a depth map is predicted using a pre-trained AI model. Finally, a 3D mesh is created using the image and depth data, and exported as a 3D model file with color mapping for basic visualization.

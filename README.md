# Real-Time Emotion Detection

## 📌 Setup Instructions
1. **Extract the ZIP file** to a folder.
2. **Open the folder in VS Code**.
3. **Install Dependencies** by running:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Program**:
   ```bash
   python main.py
   ```
5. **Press 'q'** to quit the webcam window.

## 📌 Features
- Detects **text-based emotions** using NLTK.
- Detects **facial emotions** using OpenCV.
- Simple and modular codebase.

## 📌 Notes
- The facial emotion module currently only detects faces, not specific emotions. To improve it, use **DeepFace** or **CNN models**.
- Make sure your webcam is working before running `main.py`.

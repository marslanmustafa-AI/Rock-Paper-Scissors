# Rock-Paper-Scissors Deep Learning Classifier

A full-stack AI-powered web application for classifying hand gestures into Rock, Paper, or Scissors. This project uses a custom-trained **YOLO (Ultralytics)** classification model served via a **FastAPI** backend, and provides a rich, interactive **Next.js** frontend for live webcam predictions, camera snapshots, and image uploads.

## ✨ Features

- **Live Mode**: Real-time Rock-Paper-Scissors classification using your webcam. The app continuously polls the video feed and provides debounced, smooth prediction updates.
- **Camera Snapshot**: Take a picture directly from your webcam to get an instant prediction.
- **Image Upload**: Upload any image of a hand gesture from your local device to classify it.
- **Detailed Analytics**: View confidence scores for all classes and an extended "Top 5 Predictions" view supported by the YOLO model.
- **Modern UI**: A responsive, dark-themed user interface built with Next.js, React hooks, and Lucide React icons.

## 📂 Project Structure

- **`api/`**: Contains the FastAPI backend application. It loads the pre-trained YOLO model and exposes a RESTful API (`/predict`) to process image bytes and return confidence scores.
- **`frontend/`**: The Next.js client application. It handles webcam streams, canvas drawing for frame captures, background removal (optional), and interacts with the FastAPI backend.
- **`notebooks/`**: Jupyter notebooks demonstrating the deep learning experiments. Contains training scripts for models built from scratch (`from_scratch/`) and using Ultralytics YOLO (`yolo/`).
- **`pyproject.toml`**: Python project configuration, utilizing `uv` for ultra-fast dependency management.

## 🚀 Getting Started

### Prerequisites

- Node.js (v18+)
- Python (3.11+)
- `uv` (recommended) or `pip` for Python dependency management

### 1. Running the Backend (FastAPI)

Navigate to the project root and install the required Python dependencies:

```bash
# If using uv
uv venv
source .venv/bin/activate
uv pip install -e .

# Or using pip
pip install fastapi uvicorn ultralytics python-multipart pillow opencv-python
```

Start the FastAPI server:

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. You can check the documentation at `http://localhost:8000/docs`.

### 2. Running the Frontend (Next.js)

Open a new terminal window, navigate to the `frontend` directory, and install dependencies:

```bash
cd frontend
npm install
# or
pnpm install
```

Start the Next.js development server:

```bash
npm run dev
# or
pnpm dev
```

The application will be accessible at `http://localhost:3000`.

## 🧠 Technologies Used

- **Deep Learning**: Ultralytics (YOLO), Keras, TensorFlow, OpenCV, NumPy
- **Backend**: FastAPI, Pydantic, Python
- **Frontend**: Next.js, React, TypeScript, CSS (Vanilla), Lucide React
- **Package Management**: npm/pnpm (Frontend), uv (Backend)

---

### Author
**Muhammad Arslan**  
📧 Email: [marslanmustafa391@gmail.com](mailto:marslanmustafa391@gmail.com)  
🌐 Website: [https://marslanmustafa.com](https://marslanmustafa.com)

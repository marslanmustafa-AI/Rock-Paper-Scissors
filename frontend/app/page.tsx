'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Hand, HandMetal, Scissors, Upload, AlertCircle, Camera, Play, Square, Loader2 } from 'lucide-react';
import { removeBackground } from '@imgly/background-removal';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
// const API_URL = 'http://192.168.18.43:8000';

interface PredictionResult {
  predicted_class: string;
  confidence: number;
  all_predictions: Record<string, number>;
  model_type?: string;
  top5_classes?: string[];
  top5_confidences?: number[];
}

interface ApiInfo {
  model: string;
  model_type: string;
  labels: string[];
}

type Mode = 'idle' | 'upload' | 'camera' | 'live';

const icons: Record<string, React.ComponentType<{ size: number }>> = {
  rock: HandMetal,
  paper: Hand,
  scissors: Scissors,
};

export default function Home() {
  const [currentMode, setCurrentMode] = useState<Mode>('idle');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [processedUrl, setProcessedUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessingBg, setIsProcessingBg] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showVideo, setShowVideo] = useState(false);
  const [modelType, setModelType] = useState<string>('keras');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const liveVideoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const liveIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const cooldownRef = useRef<boolean>(false);
  const [debouncedPrediction, setDebouncedPrediction] = useState<PredictionResult | null>(null);

  // Debounce the prediction updates for a smoother UI in live mode
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedPrediction(prediction);
    }, 300); // Increased debounce time for smoother UI
    return () => clearTimeout(timer);
  }, [prediction]);

  // Fetch API info on mount to determine model type
  useEffect(() => {
    fetch(`${API_URL}/`)
      .then(res => res.json())
      .then((data: ApiInfo) => {
        setModelType(data.model_type || 'keras');
      })
      .catch(() => {
        setModelType('keras');
      });
  }, []);

  const stopCapture = useCallback(() => {
    if (liveIntervalRef.current) {
      clearInterval(liveIntervalRef.current);
      liveIntervalRef.current = null;
    }
    cooldownRef.current = false;
  }, []);

  const stopCameraStream = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    setShowVideo(false);
  }, []);

  const stopAll = useCallback(() => {
    stopCapture();
    stopCameraStream();
  }, [stopCapture, stopCameraStream]);

  useEffect(() => {
    return () => {
      stopAll();
    };
  }, [stopAll]);

  const processAndUpload = async (file: File) => {
    setIsProcessingBg(true);
    setIsLoading(true);
    setPreviewUrl(URL.createObjectURL(file));
    setProcessedUrl(null);
    setPrediction(null);

    try {
      // const blob = await removeBackground(file);
      // const processedFile = new File([blob], 'processed.png', { type: 'image/png' });
      // setProcessedUrl(URL.createObjectURL(processedFile));
      await uploadFile(file);
    } catch (err) {
      console.error('Background removal failed:', err);
      await uploadFile(file);
    } finally {
      setIsProcessingBg(false);
      setIsLoading(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const input = e.target;
    if (input.files && input.files.length > 0) {
      processAndUpload(input.files[0]);
    }
  };

  const uploadFile = async (file: File) => {
    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Prediction failed');
      setPrediction(await res.json());
    } catch {
      setError('Failed to get prediction. Is the API running?');
    } finally {
      // Throttle further captures for a cooldown period after results are fetched
      cooldownRef.current = true;
      setIsLoading(false);
      setTimeout(() => {
        cooldownRef.current = false;
      }, 800); // 800ms cooldown after fetch
    }
  };

  const getBarWidth = (confidence: number): string => {
    return `${Math.max(confidence, 5)}%`;
  };

  const reset = () => {
    stopAll();
    setPreviewUrl(null);
    setProcessedUrl(null);
    setPrediction(null);
    setError(null);
    setCurrentMode('idle');
    setShowVideo(false);
  };

  const captureAndPredict = () => {
    // Return early if we are currently loading or in a cooldown period
    if (!liveVideoRef.current || !canvasRef.current || isLoading || cooldownRef.current) return;

    const video = liveVideoRef.current;
    if (video.readyState < 2) return;

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob: Blob | null) => {
        if (blob) {
          const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
          uploadFile(file); // Call upload directly, which manages isLoading and cooldown
        }
      }, 'image/jpeg', 0.8);
    }
  };

  const initCamera = (): Promise<boolean> => {
    return new Promise((resolve) => {
      stopCapture(); // Only stop the capture loops, keep the stream alive

      if (mediaStreamRef.current && mediaStreamRef.current.active) {
        setShowVideo(true);

        const attachStream = () => {
          if (liveVideoRef.current) {
            if (liveVideoRef.current.srcObject !== mediaStreamRef.current) {
              liveVideoRef.current.srcObject = mediaStreamRef.current;
              liveVideoRef.current.onloadedmetadata = () => {
                liveVideoRef.current?.play().catch(e => console.error('Play error:', e));
                resolve(true);
              };
            } else {
              liveVideoRef.current?.play().catch(e => console.error('Play error:', e));
              resolve(true);
            }
          } else {
            setTimeout(attachStream, 50);
          }
        };
        attachStream();
        return;
      }

      setShowVideo(false);

      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          mediaStreamRef.current = stream;
          setShowVideo(true);

          const attachStream = () => {
            if (liveVideoRef.current) {
              liveVideoRef.current.srcObject = stream;
              liveVideoRef.current.onloadedmetadata = () => {
                liveVideoRef.current?.play().catch(e => console.error('Play error:', e));
                resolve(true);
              };
            } else {
              setTimeout(attachStream, 50);
            }
          };
          attachStream();
        })
        .catch(err => {
          console.error('Camera error:', err);
          setError('Camera not available. Please allow camera access.');
          setCurrentMode('idle');
          resolve(false);
        });
    });
  };

  const startLiveCapture = () => {
    if (liveIntervalRef.current) clearInterval(liveIntervalRef.current);
    liveIntervalRef.current = setInterval(captureAndPredict, 300); // Faster polling for live feel
  };

  const openCamera = async () => {
    setCurrentMode('camera');
    setPrediction(null);
    setError(null);
    const success = await initCamera();
    if (!success) {
      stopAll();
      setCurrentMode('idle');
    }
  };

  const startLiveMode = async () => {
    setCurrentMode('live');
    setPrediction(null);
    setError(null);
    const success = await initCamera();
    if (success) {
      startLiveCapture();
    } else {
      stopAll();
      setCurrentMode('idle');
    }
  };

  const stopCamera = () => {
    stopAll();
    setCurrentMode('idle');
  };

  const captureImage = () => {
    if (!liveVideoRef.current || !canvasRef.current) return;

    const video = liveVideoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob: Blob | null) => {
        if (blob) {
          const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
          stopAll();
          setShowVideo(false);
          setPreviewUrl(URL.createObjectURL(file));
          setProcessedUrl(null);
          setCurrentMode('upload');
          processAndUpload(file);
        }
      }, 'image/jpeg');
    }
  };

  const stopLive = () => {
    stopAll();
    setShowVideo(false);
    setCurrentMode('idle');
  };

  return (
    <div className="app-container">
      <div className="main-container">
        <div className="content-wrapper">
          <header className="header">
            <h1 className="title">RPS Classifier</h1>
            <p className="subtitle">Rock Paper Scissors Image Recognition</p>
          </header>

          <div className="mode-selector">
            <button
              className={`mode-btn ${currentMode === 'idle' || currentMode === 'upload' ? 'active' : ''}`}
              onClick={reset}
            >
              <Upload size={20} />
              Upload
            </button>
            <button
              className={`mode-btn ${currentMode === 'camera' ? 'active' : ''}`}
              onClick={openCamera}
            >
              <Camera size={20} />
              Camera
            </button>
            <button
              className={`mode-btn live-btn ${currentMode === 'live' ? 'active' : ''}`}
              onClick={startLiveMode}
            >
              <Play size={20} />
              Live
            </button>
          </div>

          {showVideo && (
            <div className="video-container">
              <div className="video-wrapper">
                <video ref={liveVideoRef} className="live-video" playsInline autoPlay muted></video>
                <canvas ref={canvasRef} hidden></canvas>
                {isLoading && currentMode === 'live' && (
                  <div className="live-loader">
                    <Loader2 size={32} className="spin-icon" />
                  </div>
                )}
              </div>

              {currentMode === 'camera' && (
                <div className="camera-controls">
                  <button className="cancel-btn" onClick={stopCamera}>
                    <Square size={18} />
                    Cancel
                  </button>
                  <button className="capture-btn-large" onClick={captureImage}>
                    <Camera size={24} />
                  </button>
                </div>
              )}

              {currentMode === 'live' && (
                <div className="live-controls">
                  <button className="stop-btn" onClick={stopLive}>
                    <Square size={18} />
                    Stop
                  </button>
                  <button className="capture-btn" onClick={captureImage}>
                    <Camera size={18} />
                    Capture
                  </button>
                </div>
              )}
            </div>
          )}

          {(currentMode === 'idle' || currentMode === 'upload') && (
            <div className="card">
              <div className="preview-container">
                {processedUrl || previewUrl ? (
                  <img src={processedUrl ?? previewUrl ?? ''} className="preview-image" alt="Preview" />
                ) : (
                  <div className="empty-state" onClick={() => fileInputRef.current?.click()}>
                    <Upload size={48} className="empty-icon" />
                    <p className="empty-text">Upload an image to classify</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {isProcessingBg && (
            <div className="loading">
              <Loader2 size={24} className="spin-icon" />
              <p>Removing background...</p>
            </div>
          )}
          {isLoading && currentMode !== 'live' && !isProcessingBg && (
            <div className="loading">
              <Loader2 size={24} className="spin-icon" />
              <p>Analyzing image...</p>
            </div>
          )}

          {error && !showVideo && (
            <div className="error-message">
              <AlertCircle size={20} />
              {error}
            </div>
          )}

          {debouncedPrediction && (!isLoading || currentMode === 'live') && (
            <div className="result-card">
              <div className="result-main">
                {debouncedPrediction.predicted_class && (() => {
                  const IconComponent = icons[debouncedPrediction.predicted_class];
                  return IconComponent ? (
                    <span className="result-icon">
                      <IconComponent size={56} />
                    </span>
                  ) : null;
                })()}
                <div className="result-info">
                  <span className="result-class">{debouncedPrediction.predicted_class.toUpperCase()}</span>
                  <span className="result-confidence">{debouncedPrediction.confidence.toFixed(1)}%</span>
                </div>
              </div>

              {/* Show extended YOLO data if model_type is yolo */}
              {modelType === 'yolo' && debouncedPrediction.top5_classes && (
                <div className="top5-section">
                  <h4 className="top5-title">Top 5 Predictions</h4>
                  <div className="top5-list">
                    {debouncedPrediction.top5_classes.map((cls, idx) => {
                      const IconComponent = icons[cls];
                      return (
                        <div key={cls} className="top5-item">
                          <span className="top5-rank">#{idx + 1}</span>
                          {IconComponent && <IconComponent size={16} />}
                          <span className="top5-label">{cls}</span>
                          <span className="top5-conf">{debouncedPrediction.top5_confidences?.[idx].toFixed(1)}%</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              <div className="scores-grid">
                {Object.entries(debouncedPrediction.all_predictions).map(([label, conf]) => {
                  const IconComponent = icons[label];
                  return (
                    <div key={label} className="score-item">
                      <div className="score-header">
                        {IconComponent && <IconComponent size={16} />}
                        <span>{label}</span>
                      </div>
                      <div className="score-bar">
                        <div
                          className="score-fill"
                          style={{
                            width: getBarWidth(conf),
                            background: debouncedPrediction.predicted_class === label ? 'linear-gradient(90deg, #6366f1, #8b5cf6)' : 'rgba(255, 255, 255, 0.2)'
                          }}
                        ></div>
                      </div>
                      <span className="score-value">{conf.toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {(currentMode === 'idle' || currentMode === 'upload') && (
            <div className="actions">
              <button className="action-btn primary" onClick={() => fileInputRef.current?.click()}>
                <Upload size={18} />
                Choose File
              </button>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={handleFileSelect}
          />
        </div>
      </div>
    </div>
  );
}

import { useState, useRef } from "react";
import PropTypes from "prop-types";

// Placeholder assets since we can't import actual files
const PlusIcon = () => (
  <svg className="w-7 h-7" fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M10 3a1 1 0 011 1v5h5a1 1 0 110 2h-5v5a1 1 0 11-2 0v-5H4a1 1 0 110-2h5V4a1 1 0 011-1z" clipRule="evenodd" />
  </svg>
);

const Nav = () => (
  <nav className="container mx-auto px-4 py-4">
    <h1 className="text-xl font-bold text-white">AI Deepfake Detector</h1>
  </nav>
);

// File upload container
const FileUploadButton = ({ type, onFileChange, acceptedFormats }) => {
  const fileInputRef = useRef(null);

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      onFileChange(file);
    }
  };

  return (
    <div className="text-white bg-slate-900 h-60 max-w-screen-2xl font-bold rounded-3xl flex items-center justify-center flex-col my-5">
      <button
        className="bg-blue-600 rounded-xl my-3 h-16 w-56 flex items-center justify-center space-x-3 hover:bg-blue-700 transition-colors"
        onClick={handleUploadClick}
      >
        <PlusIcon />
        <span>Upload {type}</span>
      </button>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: "none" }}
        accept={acceptedFormats
          .map((format) => `${type === "Image" ? "image/" : "video/"}${format}`)
          .join(",")}
      />
      <span className="text-sm">Drop a {type.toLowerCase()} or click to upload</span>

      <div className="flex items-center justify-center flex-row text-sm space-x-3 pt-5">
        <div>Supported formats:</div>
        {acceptedFormats.map((format) => (
          <div
            key={format}
            className="border-gray-600 border-2 bg-gray-800 rounded-lg px-2"
          >
            {format}
          </div>
        ))}
      </div>
    </div>
  );
};

FileUploadButton.propTypes = {
  type: PropTypes.oneOf(["Image", "Video"]).isRequired,
  onFileChange: PropTypes.func.isRequired,
  acceptedFormats: PropTypes.arrayOf(PropTypes.string).isRequired,
};

// Main App component
const App = () => {
  const [activeTab, setActiveTab] = useState("Image");
  const [file, setFile] = useState(null);
  const [fileURL, setFileURL] = useState(null);
  const [processedMedia, setProcessedMedia] = useState(null);
  const [detectionResults, setDetectionResults] = useState(null);
  const [videoResults, setVideoResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileUpload = (file) => {
    const url = URL.createObjectURL(file);
    setFileURL(url);
    setFile(file);
    setProcessedMedia(null);
    setDetectionResults(null);
    setVideoResults(null);
    setError(null);
  };

  const clearUpload = () => {
    setFile(null);
    setFileURL(null);
    setProcessedMedia(null);
    setDetectionResults(null);
    setVideoResults(null);
    setError(null);
  };

  // Image DeepFake Detection Function
  const handleDetectImageDeepfake = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    const reader = new FileReader();
    reader.onload = async () => {
      const base64Image = reader.result;

      try {
        const response = await fetch("http://localhost:8000/api/detect-image/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            image: base64Image,
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          console.error("Server error:", errorData);
          setError("Error processing image: " + (errorData.error || "Unknown error"));
          setIsLoading(false);
          return;
        }

        const data = await response.json();

        if (data.faces_detected === 0) {
          setError("No faces detected in the image");
          setIsLoading(false);
          return;
        }

        // Handle the response from Django API
        setProcessedMedia(data.processed_image);
        setDetectionResults(data.results);
      } catch (error) {
        console.error("Error detecting deepfake:", error);
        setError("Error processing image: " + error.message);
      } finally {
        setIsLoading(false);
      }
    };

    reader.readAsDataURL(file);
  };

  // Video DeepFake Detection Function
  const handleDetectVideoDeepfake = async () => {
    if (!file) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("video", file);

    try {
      const response = await fetch("http://localhost:8000/api/detect-video/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Server error:", errorData);
        setError("Error processing video: " + (errorData.error || "Unknown error"));
        setIsLoading(false);
        return;
      }

      const data = await response.json();
      console.log("Video detection response:", data);

      // Check for faces in the video based on Django response structure
      const detectionResults = data.detection_results;
      
      // Check if percentage method exists and has frames with faces
      if (detectionResults?.percentage_method?.frames_with_faces === 0) {
        setError("No faces detected in the video");
        setIsLoading(false);
        return;
      }

      setProcessedMedia(data.processed_video);
      setVideoResults(data.detection_results);
    } catch (error) {
      console.error("Error detecting deepfake in video:", error);
      setError("Error processing video: " + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDetect = () => {
    if (activeTab === "Image") {
      handleDetectImageDeepfake();
    } else {
      handleDetectVideoDeepfake();
    }
  };

  // Render video results with the Django backend response structure
  const renderVideoResults = () => {
    if (!videoResults) return null;

    const finalDecision = videoResults.final_decision;
    const percentageMethod = videoResults.percentage_method;
    const statisticalMethod = videoResults.statistical_method;
    const temporalMethod = videoResults.temporal_method;
    const clusteringMethod = videoResults.clustering_method;
    const lstmMethod = videoResults.LSTM_method;

    return (
      <div className="w-full max-w-md space-y-4">
        <h3 className="text-2xl font-bold mb-4 text-center">
          Video Detection Results
        </h3>

        {/* Main decision box */}
        <div
          className={`p-4 rounded-lg shadow-lg ${finalDecision.is_deepfake
              ? "bg-red-500/30 border-2 border-red-500"
              : "bg-green-500/30 border-2 border-green-500"
            }`}
        >
          <div className="flex justify-between items-center mb-2">
            <span className="font-bold text-lg">Final Analysis</span>
            <span
              className={`font-semibold ${finalDecision.is_deepfake ? "text-red-400" : "text-green-400"
                }`}
            >
              {finalDecision.is_deepfake ? "Likely Deepfake" : "Likely Real"}
            </span>
          </div>
          <div className="space-y-2 text-sm">
            <div>Confidence: {finalDecision.confidence?.toFixed(2)}%</div>
            <div>Method Agreement: {finalDecision.method_agreement}</div>
          </div>
        </div>

        {/* Percentage method results -------------------------------------------------------------------------------------------------------------------------------*/}
        {percentageMethod && (
          <div className="p-4 rounded-lg bg-slate-800/70 shadow-lg">
            <div className="flex justify-between items-center mb-2">
              <span className="font-bold">Frame Analysis</span>
              <span className={`text-sm ${percentageMethod.is_deepfake ? "text-red-400" : "text-green-400"}`}>
                {percentageMethod.is_deepfake ? "Deepfake" : "Real"}
              </span>
            </div>
            <div className="space-y-1 text-sm">
              <div>Frames with faces: {percentageMethod.frames_with_faces}</div>
              <div>Deepfake frames: {percentageMethod.frames_with_deepfakes}</div>
              <div>Deepfake ratio: {percentageMethod.deepfake_percentage?.toFixed(2)}%</div>
            </div>
          </div>
        )}

        {/* Statistical method results ---------------------------------------------------------------------------------------------------------------------------------------------*/}
        {statisticalMethod && (
          <div className="p-4 rounded-lg bg-slate-800/70 shadow-lg">
            <div className="flex justify-between items-center mb-2">
              <span className="font-bold">Statistical Analysis</span>
              <span className={`text-sm ${statisticalMethod.is_deepfake ? "text-red-400" : "text-green-400"}`}>
                {statisticalMethod.is_deepfake ? "Deepfake" : "Real"}
              </span>
            </div>
            <div className="space-y-1 text-sm">
              <div>Mean confidence: {statisticalMethod.mean_confidence?.toFixed(3)}</div>
              <div>Weighted confidence: {statisticalMethod.weighted_confidence?.toFixed(3)}</div>
              <div>Standard deviation: {statisticalMethod.std_deviation?.toFixed(3)}</div>
            </div>
          </div>
        )}

        {/* Temporal method results--------------------------------------------------------------------------------------------------------------------------------------- */}
        {temporalMethod && (
          <div className="p-4 rounded-lg bg-slate-800/70 shadow-lg">
            <div className="flex justify-between items-center mb-2">
              <span className="font-bold">Temporal Analysis</span>
              <span className={`text-sm ${temporalMethod.is_deepfake ? "text-red-400" : "text-green-400"}`}>
                {temporalMethod.is_deepfake ? "Deepfake" : "Real"}
              </span>
            </div>
            <div className="space-y-1 text-sm">
              <div>Avg confidence change: {temporalMethod.avg_confidence_change?.toFixed(3)}</div>
              <div>Max confidence change: {temporalMethod.max_confidence_change?.toFixed(3)}</div>
              <div>Sustained pattern: {temporalMethod.sustained_pattern?.toFixed(3)}</div>
            </div>
          </div>
        )}

        {/* Clustering method results -------*/}
        {clusteringMethod && (
          <div className="p-4 rounded-lg bg-slate-800/70 shadow-lg">
            <div className="flex justify-between items-center mb-2">
              <span className="font-bold">Clustering Analysis</span>
              <span className={`text-sm ${clusteringMethod.is_deepfake ? "text-red-400" : "text-green-400"}`}>
                {clusteringMethod.is_deepfake ? "Deepfake" : "Real"}
              </span>
            </div>
            <div className="space-y-1 text-sm">
              <div>Fake cluster %: {clusteringMethod.fake_cluster_percentage?.toFixed(2)}%</div>
              <div>Cluster means: [{clusteringMethod.cluster_means?.map(m => m.toFixed(3)).join(', ')}]</div>
            </div>
          </div>
        )}

        {/* LSTM method results ---------------------------------------------------------------------------------------------------------------------*/}
        {lstmMethod && (
          <div className="p-4 rounded-lg bg-slate-800/70 shadow-lg">
            <div className="flex justify-between items-center mb-2">
              <span className="font-bold">LSTM Analysis</span>
              <span className={`text-sm ${lstmMethod.is_deepfake ? "text-red-400" : "text-green-400"}`}>
                {lstmMethod.is_deepfake ? "Deepfake" : "Real"}
              </span>
            </div>
            <div className="space-y-1 text-sm">
              <div>LSTM Confidence: {(lstmMethod.Confidence_LSTM * 100)?.toFixed(2)}%</div>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white">
      <header className="bg-black/30 shadow-md">
        <Nav />
      </header>

      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-center mb-8 text-white drop-shadow-lg">
          Deepfake Detection AI
        </h1>

        {/* Tab buttons */}
        <div className="flex justify-center gap-4 mb-6">
          <button
            className={`px-6 py-2 rounded-lg font-bold transition-colors ${activeTab === "Image"
                ? "bg-blue-600 text-white"
                : "bg-slate-700 hover:bg-slate-600"
              }`}
            onClick={() => {
              setActiveTab("Image");
              clearUpload();
            }}
          >
            Image Detection
          </button>
          <button
            className={`px-6 py-2 rounded-lg font-bold transition-colors ${activeTab === "Video"
                ? "bg-blue-600 text-white"
                : "bg-slate-700 hover:bg-slate-600"
              }`}
            onClick={() => {
              setActiveTab("Video");
              clearUpload();
            }}
          >
            Video Detection
          </button>
        </div>

        {/* Upload area */}
        <div className="grid md:grid-cols-1 gap-8">
          {activeTab === "Image" ? (
            <FileUploadButton
              type="Image"
              onFileChange={handleFileUpload}
              acceptedFormats={["png", "jpeg", "jpg", "heic"]}
            />
          ) : (
            <FileUploadButton
              type="Video"
              onFileChange={handleFileUpload}
              acceptedFormats={["mp4", "avi", "mov"]}
            />
          )}
        </div>

        {/* Results area */}
        {(fileURL || processedMedia) && (
          <div className="bg-black/50 rounded-2xl p-6 mt-8 space-y-6">
            <div className="flex justify-center space-x-4">
              {file && (
                <>
                  <button
                    className="bg-red-600 hover:bg-red-700 rounded-lg text-white font-bold px-6 py-2 transition-colors"
                    onClick={clearUpload}
                    disabled={isLoading}
                  >
                    Clear
                  </button>
                  <button
                    className={`bg-green-700 hover:bg-green-900 rounded-lg text-white font-bold px-6 py-2 transition-colors ${isLoading ? "opacity-50 cursor-not-allowed" : ""
                      }`}
                    onClick={handleDetect}
                    disabled={isLoading}
                  >
                    {isLoading ? "Processing..." : "Detect Deepfake"}
                  </button>
                </>
              )}
            </div>

            {error && (
              <div className="bg-red-500/30 border-2 border-red-500 p-4 rounded-lg text-center">
                {error}
              </div>
            )}

            <div className="flex flex-col md:flex-row gap-8 items-center justify-center">
              <div className="w-full max-w-xl">
                {activeTab === "Image" ? (
                  <div className="flex flex-col gap-4">
                    {fileURL && !processedMedia && (
                      <div>
                        <h4 className="text-lg font-semibold mb-2">Original Image</h4>
                        <img
                          src={fileURL}
                          alt="Original Upload"
                          className="rounded-lg shadow-lg max-h-[300px] w-full object-contain"
                        />
                      </div>
                    )}
                    {processedMedia && (
                      <div>
                        <h4 className="text-lg font-semibold mb-2">Processed Image</h4>
                        <img
                          src={processedMedia}
                          alt="Processed"
                          className="rounded-lg shadow-lg max-h-[300px] w-full object-contain"
                        />
                      </div>
                    )}
                  </div>
                ) : (
                  <div>
                    {processedMedia ? (
                      <>
                        <h4 className="text-lg font-semibold mb-2">Processed Video</h4>
                        <video
                          src={processedMedia}
                          controls
                          className="rounded-lg shadow-2xl max-h-[400px] w-full"
                        />
                      </>
                    ) : fileURL ? (
                      <>
                        <h4 className="text-lg font-semibold mb-2">Original Video</h4>
                        <video
                          src={fileURL}
                          controls
                          className="rounded-lg shadow-2xl max-h-[400px] w-full"
                        />
                      </>
                    ) : null}
                  </div>
                )}
              </div>

              {activeTab === "Image" && detectionResults && (
                <div className="w-full max-w-md space-y-4">
                  <h3 className="text-2xl font-bold mb-4 text-center">
                    Detection Results
                  </h3>
                  {detectionResults.map((result, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg shadow-lg ${result.is_deepfake
                          ? 'bg-red-500/30 border-2 border-red-500'
                          : 'bg-green-500/30 border-2 border-green-500'
                        }`}
                    >
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-bold text-lg">
                          Face {index + 1}
                        </span>
                        <span className={`font-semibold ${result.is_deepfake ? 'text-red-400' : 'text-green-400'
                          }`}>
                          {result.is_deepfake ? "Deepfake" : "Real"}
                        </span>
                      </div>
                      <div className="space-y-1 text-sm">
                        <div>Confidence: {(result.confidence * 100).toFixed(2)}%</div>
                        <div>Position: [{result.face_coordinates.join(', ')}]</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {activeTab === "Video" && videoResults && renderVideoResults()}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;
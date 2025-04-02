import { useState, useRef } from "react";
import PropTypes from "prop-types";
import Nav from "./Components/Nav";
import Plus from "./assets/Plus.png";
import Picture from "./assets/AI.jpeg";
// file uplaod container---------------------------------------------------------------------------------------------------------------------
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
        className="bg-blue-600 rounded-xl my-3 h-16 w-56 flex items-center justify-center space-x-3"
        onClick={handleUploadClick}
      >
        <img src={Plus} alt="Plus Icon" className="w-7" />
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
      <span className="text-sm">Drop a {type.toLowerCase()} or Paste URL</span>

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
// Add PropTypes validation------------------------------------------------------------------------------------------------
FileUploadButton.propTypes = {
  type: PropTypes.oneOf(["Image", "Video"]).isRequired,
  onFileChange: PropTypes.func.isRequired,
  acceptedFormats: PropTypes.arrayOf(PropTypes.string).isRequired,
};
//The main code---------------------------------------------------------------------------------------------------------------
const App = () => {
  const [imageFile, setImageFile] = useState(null);
  const [imageURL, setImageURL] = useState(Picture);
  const [processedImage, setProcessedImage] = useState(null);
  const [detectionResults, setDetectionResults] = useState(null);

  const handleImageUpload = (file) => {
    const url = URL.createObjectURL(file);
    setImageURL(url);
    setImageFile(file);
    setProcessedImage(null);
    setDetectionResults(null);
  };

  const clearUpload = () => {
    setImageFile(null);
    setImageURL(Picture);
    setProcessedImage(null);
    setDetectionResults(null);
  };
  // The DeepFake Detection Function-------------------------------------------------------------------------------------------------------------------------------------------
  const handleDetectDeepfake = async () => {
    if (!imageFile) return;

    const reader = new FileReader();
    reader.onload = async () => {
      const base64Image = reader.result; // Don't split here
      console.log("Sending image data:", base64Image.substring(0, 100)); // Log first 100 chars

      try {
        const response = await fetch("http://localhost:8000/api/detect/", {
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
          return;
        }

        const data = await response.json();

        if (data.faces_detected === 0) {
          alert("No faces detected in the image");
          return;
        }

        setProcessedImage(data.processed_image);
        setDetectionResults(data.results);
      } catch (error) {
        console.error("Error detecting deepfake:", error);
        alert("Error processing image");
      }
    };

    reader.readAsDataURL(imageFile);
  };
  //The main state----------------------------------------------------------------------------------------------------
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white">
      <header className="bg-black/30 shadow-md">
        <Nav />
      </header>
      
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-center mb-8 text-white drop-shadow-lg">
          Deepfake Detection AI
        </h1>

        <div className="grid md:grid-cols-2 gap-8">
          <FileUploadButton
            type="Image"
            onFileChange={handleImageUpload}
            acceptedFormats={["png", "jpeg", "jpg", "heic"]}
          />
          <FileUploadButton
            type="Video"
            onFileChange={handleImageUpload}
            acceptedFormats={["mp4", "avi", "mov"]}
          />
        </div>

        {(imageURL || processedImage) && (
          <div className="bg-black/50 rounded-2xl p-6 mt-8 space-y-6">
            <div className="flex justify-center space-x-4">
              {imageFile && (
                <>
                  <button
                    className="bg-red-600 hover:bg-red-700 rounded-lg text-white font-bold px-6 py-2 transition-colors"
                    onClick={clearUpload}
                  >
                    Clear
                  </button>
                  <button
                    className="bg-green-700 hover:bg-green-900 rounded-lg text-white font-bold px-6 py-2 transition-colors"
                    onClick={handleDetectDeepfake}
                  >
                    Detect Deepfake
                  </button>
                </>
              )}
            </div>

            <div className="flex flex-col md:flex-row gap-8 items-center justify-center">
              <div className="w-full max-w-xl">
                <img
                  src={processedImage || imageURL}
                  alt="Preview"
                  className="rounded-lg shadow-2xl max-h-[500px] w-full object-contain"
                />
              </div>

              {detectionResults && (
                <div className="w-full max-w-md space-y-4">
                  <h3 className="text-2xl font-bold mb-4 text-center">
                    Detection Results
                  </h3>
                  {detectionResults.map((result, index) => (
                    <div
                      key={index}
                      className={`p-4 rounded-lg shadow-lg ${
                        result.is_deepfake 
                          ? 'bg-red-500/30 border-2 border-red-500' 
                          : 'bg-green-500/30 border-2 border-green-500'
                      }`}
                    >
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-bold text-lg">
                          Face {index + 1}
                        </span>
                        <span className={`font-semibold ${
                          result.is_deepfake ? 'text-red-800' : 'text-green-600'
                        }`}>
                          {result.is_deepfake ? "Deepfake" : "Real"}
                        </span>
                      </div>
                      <div className="text-sm">
                        Confidence: {(result.confidence * 100).toFixed(2)}%
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};
export default App;

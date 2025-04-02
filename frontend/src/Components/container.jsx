import {} from 'react';
import Plus from "./assets/Plus.png";

const container = () => {
    return (
        <div className="text-2xl font-semibold justify-center items-center px-10">
          Detecte the deepfake in the video
          <div className="text-white bg-slate-900 h-60 max-w-screen-2xl font-bold rounded-3xl flex items-center justify-center flex-col my-5">
            <button className="bg-blue-600 rounded-xl my-3 h-16 w-56 flex items-center justify-center space-x-3">
              <img src={Plus} alt="Plus Icon" className="w-7 " />
              <span className="">Upload Video</span>
            </button>
            <span className="text-sm">Drop an Video or Paste URL</span>

            <div className="flex items-center justify-center flex-row text-sm space-x-3 pt-5">
              <div>Supported formats:</div>
              <div className="border-gray-600 border-2 bg-gray-800 rounded-lg px-2">
                mp4
              </div>
            </div>
          </div>
        </div>
    );
};

export default container;
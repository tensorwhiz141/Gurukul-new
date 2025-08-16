import React, { createContext, useContext, useState } from 'react';

const VideoContext = createContext();

export const useVideo = () => {
  const context = useContext(VideoContext);
  if (!context) {
    throw new Error('useVideo must be used within a VideoProvider');
  }
  return context;
};

export const VideoProvider = ({ children }) => {
  const [generatedVideo, setGeneratedVideo] = useState(null);
  const [showVideoInSidebar, setShowVideoInSidebar] = useState(false);

  const showVideo = (videoData) => {
    setGeneratedVideo(videoData);
    setShowVideoInSidebar(true);
  };

  const hideVideo = () => {
    setShowVideoInSidebar(false);
    setGeneratedVideo(null);
  };

  return (
    <VideoContext.Provider value={{
      generatedVideo,
      showVideoInSidebar,
      showVideo,
      hideVideo
    }}>
      {children}
    </VideoContext.Provider>
  );
};

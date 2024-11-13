import React, { useState } from 'react';
import axios from 'axios';

function Upload() {
  let api = "https://dog-suitable-visually.ngrok-free.app";
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [patchedImage, setPatchedImage] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setUploadedImage(URL.createObjectURL(e.target.files[0]));
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage('Please select an image to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
      // Upload the image
      const uploadResponse = await axios.post(`${api}/upload-image`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          "ngrok-skip-browser-warning": "true",
        }
      });
      
      setMessage(uploadResponse.data.message);
      
      // Extract image path from upload response
      const imageUrlPath = uploadResponse.data.image_url;
      if (imageUrlPath) {
        setImageUrl(api + imageUrlPath); // Adjust if `image_url` is a relative path
      }

      // Send the image path to get the patched image
      const patchResponse = await axios.post(`${api}/get-patch`, {
        headers: {
          'Content-Type': 'multipart/form-data',
          "ngrok-skip-browser-warning": "true",
        },
        image_url: '.'+imageUrlPath
      }, {
        responseType: 'blob' // Expecting an image blob as response
      });

      // Convert blob response to URL
      const patchedImageUrl = URL.createObjectURL(patchResponse.data);
      setPatchedImage(patchedImageUrl);

    } catch (error) {
      console.error('Error processing image:', error);
      setMessage('Failed to upload or patch image');
    }
  };

  return (
    <div className='text-white'>
      <h2>Upload an Image</h2>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload Image</button>
      <p>{message}</p>

      <div style={{width:'50%'}} className='row'>
      
      {uploadedImage && (
        <div className='col'>
          <h3>Original Uploaded Image:</h3>
          <img src={uploadedImage} alt="Uploaded" style={{ width: "100%" }} />
        </div>
      )}

      {patchedImage && (
        <div className='col'>
          <h3>Patched Image:</h3>
          <img src={patchedImage} alt="Patched" style={{ width: "100%" }} />
        </div>
      )}
      </div>
    </div>
  );
}

export default Upload;

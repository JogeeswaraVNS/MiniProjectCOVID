import React, { useState } from 'react';
import axios from 'axios';

function Upload() {
  let api = "https://dog-suitable-visually.ngrok-free.app/";
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [imageUrl, setImageUrl] = useState('');  // State to hold the uploaded image URL

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage('Please select an image to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post(`${api}upload-image`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setMessage(response.data.message);
      
      // Set the uploaded image URL for visualization
      if (response.data.image_url) {
        setImageUrl(api + response.data.image_url); // Adjust if `image_url` is a relative path
      }
    } catch (error) {
      console.error('Error uploading image:', error);
      setMessage('Failed to upload image');
    }
  };

  return (
    <div>
      <h2>Upload an Image</h2>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload Image</button>
      <p>{message}</p>
      
      {/* Display the uploaded image */}
      {imageUrl && (
        <div>
          <h3>Uploaded Image:</h3>
          <img src={imageUrl} alt="Uploaded preview" style={{ width: '300px', marginTop: '10px' }} />
        </div>
      )}
    </div>
  );
}

export default Upload;

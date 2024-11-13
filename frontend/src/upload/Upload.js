import React, { useState } from 'react';
import axios from 'axios';

function Upload() {
  const [response, setResponse] = useState('');
  let api = "https://dog-suitable-visually.ngrok-free.app";

  const handlePredict = async () => {
    try {
      const result = await axios.post(`${api}/predict`, {
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      });
      setResponse(result.data);
    } catch (error) {
      console.error("Error making prediction request", error);
      setResponse("An error occurred");
    }
  };

  return (
    <div className='text-white'>
      <h1>Upload</h1>
      <button onClick={handlePredict}>Make Prediction</button>
      <p>Response: {response}</p>
    </div>
  );
}

export default Upload;

import React, { useState } from "react";
import axios from "axios";

function Upload() {
  let api = "https://dog-suitable-visually.ngrok-free.app";
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [gradCamImage1, setGradCamImage1] = useState(null);
  const [gradCamImage2, setGradCamImage2] = useState(null);
  const [gradCamImage3, setGradCamImage3] = useState(null);
  const [gradCamImage4, setGradCamImage4] = useState(null);
  const [patchedImage, setPatchedImage] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);

  const handleFileChange = (e) => {
    setMessage(false)
    setPatchedImage(false)
    setGradCamImage1(false);
    setGradCamImage2(false);
    setGradCamImage3(false);
    setGradCamImage4(false);
    setFile(e.target.files[0]);
    setUploadedImage(URL.createObjectURL(e.target.files[0]));
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select an image to upload.");
      return;
    }
    setMessage(false)
    setPatchedImage(false)
    setGradCamImage1(false);
    setGradCamImage2(false);
    setGradCamImage3(false);
    setGradCamImage4(false);

    const formData = new FormData();
    formData.append("image", file);

    try {
      // Upload the image
      const uploadResponse = await axios.post(`${api}/upload-image`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          "ngrok-skip-browser-warning": "true",
        },
      });

      setMessage(uploadResponse.data.message);

      // Extract image path from upload response
      const imageUrlPath = uploadResponse.data.image_url;
      if (imageUrlPath) {
        setImageUrl(api + imageUrlPath); // Adjust if `image_url` is a relative path
      }

      // Send the image path to get the patched image
      const patchResponse = await axios.post(
        `${api}/get-patch`,
        {
          headers: {
            "Content-Type": "multipart/form-data",
            "ngrok-skip-browser-warning": "true",
          },
          image_url: imageUrlPath,
        },
        {
          responseType: "blob", // Expecting an image blob as response
        }
      );

      // Convert blob response to URL
      const patchedImageUrl = URL.createObjectURL(patchResponse.data);
      setPatchedImage(patchedImageUrl);

      const response1 = await axios.post(`${api}/GradCamLayer1`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          "ngrok-skip-browser-warning": "true",
        },
        responseType: "blob",
      });

      const imageBlob1 = response1.data;
      const imageUrl1 = URL.createObjectURL(imageBlob1);

      setGradCamImage1(imageUrl1);
      const response2 = await axios.post(`${api}/GradCamLayer2`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          "ngrok-skip-browser-warning": "true",
        },
        responseType: "blob",
      });

      const imageBlob2 = response2.data;
      const imageUrl2 = URL.createObjectURL(imageBlob2);
      setGradCamImage2(imageUrl2);

      const response3 = await axios.post(`${api}/GradCamLayer3`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          "ngrok-skip-browser-warning": "true",
        },
        responseType: "blob",
      });

      const imageBlob3 = response3.data;
      const imageUrl3 = URL.createObjectURL(imageBlob3);
      setGradCamImage3(imageUrl3);

      const response4 = await axios.post(`${api}/GradCamLayer4`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
          "ngrok-skip-browser-warning": "true",
        },
        responseType: "blob",
      });

      const imageBlob4 = response4.data;
      const imageUrl4 = URL.createObjectURL(imageBlob4);
      setGradCamImage4(imageUrl4);
    } catch (error) {
      console.error("Error processing image:", error);
      setMessage("Failed to upload or patch image");
    }
  };

  return (
    <div className="text-white text-center px-5 pt-4">
      <div>
        <h2
          style={{
            fontSize: "1.5rem",
            fontWeight: "bold",
            marginBottom: "1rem",
          }}
        >
          Upload an Image
        </h2>
        <input
          className="form-control btn btn-primary"
          style={{ fontSize: "1.2rem", width: "40%" }}
          type="file"
          onChange={handleFileChange}
        />
        <br></br>
        <button
          className="btn btn-primary mt-4"
          onClick={handleUpload}
          style={{ fontSize: "1.2rem", width: "40%" }}
        >
          Upload Image
        </button>
      </div>

      <div style={{ width: "100%" }} className="row pt-5">
        {uploadedImage && (
          <div className="col-2">
            <img src={uploadedImage} alt="Uploaded" style={{ width: "100%" }} />
            {message && <h5 className={`py-2 ${message === 'Positive' ? 'bg-danger' : 'bg-success'}`}>Categorised as {message}</h5>}
          </div>
        )}

        {patchedImage && (
          <div className="col-2">
            <img src={patchedImage} alt="Patched" style={{ width: "100%" }} />
            <h5 className="pt-2">Patched Image</h5>
          </div>
        )}

        {gradCamImage1 && (
          <div className="col-2">
            <img src={gradCamImage1} alt="Grad-CAM" style={{ width: "100%" }} />
            <h5 className="pt-2">CNN Layer 1 Grad-CAM</h5>
          </div>
        )}

        {gradCamImage2 && (
          <div className="col-2">
            <img src={gradCamImage2} alt="Grad-CAM" style={{ width: "100%" }} />
            <h5 className="pt-2">CNN Layer 2 Grad-CAM</h5>
          </div>
        )}

        {gradCamImage3 && (
          <div className="col-2">
            <img src={gradCamImage3} alt="Grad-CAM" style={{ width: "100%" }} />
            <h5 className="pt-2">CNN Layer 3 Grad-CAM</h5>
          </div>
        )}

        {gradCamImage4 && (
          <div className="col-2">
            <img src={gradCamImage4} alt="Grad-CAM" style={{ width: "100%" }} />
            <h5 className="pt-2">CNN Layer 4 Grad-CAM</h5>
          </div>
        )}
      </div>
    </div>
  );
}

export default Upload;

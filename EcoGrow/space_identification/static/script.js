
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const captureButton = document.getElementById("capture-btn");
const newButton = document.getElementById("new-btn");
const submitButton = document.getElementById("submit-btn");
const predictionResultCam = document.getElementById("prediction-result-cam");
const predictionResultForm = document.getElementById("prediction-result-form");
const fileInput = document.getElementById("file");
const ctx = canvas.getContext("2d");




if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream;
        })
        .catch((error) => {
            console.error("Can't access the camera", error);
        });
}


captureButton.addEventListener("click", () => {



    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    video.style.display = "none";
    canvas.style.display = "block";

    captureButton.style.display = "none";
    newButton.style.display = "inline-block";
    submitButton.style.display = "inline-block";
});


newButton.addEventListener("click", () => {
    video.style.display = "block";
    canvas.style.display = "none";


    captureButton.style.display = "inline-block";
    newButton.style.display = "none";
    submitButton.style.display = "none";
});

submitButton.addEventListener("click", () => {

    if (predictionResultForm.innerText.trim()){
        alert("Image is already uploaded. Please Refresh to capture an image");
        return;
    }

    canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append("file", blob, "captured-image.png");

        fetch("/space_identification/", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            predictionResultCam.innerText = `Prediction: ${data.prediction}`;
        })
        .catch(error => console.error("Error:", error));
    }, "image/png");
});

document.getElementById("uploadButton").addEventListener("click", () => {
    const imageCaptured = ctx && ctx.getImageData(0, 0, canvas.width, canvas.height).data.some(channel => channel !== 0);
    //const predictionResultCam = document.getElementById("prediction-result-cam");

    if (predictionResultCam.innerText.trim()){
        alert("Image is already captured. Please reftresh to upload file");
        return;
    }
    const fileInput = document.getElementById("file");
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("/space_identification/", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        predictionResultForm.innerText = `Prediction: ${data.prediction}`;
    })
    .catch(error => console.error("Error:", error));
    });


function validation(){

    const imageCaptured = ctx && ctx.getImageData(0, 0, canvas.width, canvas.height).data.some(channel => channel !== 0);
    const fileInput = document.getElementById("file");
    const fileuploaded =  fileInput && fileInput.files.length > 0;

    if (imageCaptured || fileuploaded ){
        const predictionResultCam = document.getElementById("prediction-result-cam")?.innerText.trim();
        const predictionResultForm = document.getElementById("prediction-result-form")?.innerText.trim();

        let result = predictionResultForm || predictionResultCam;
        result = result.replace("Prediction: ", "").trim();

        
       // if (predictionResultCam && predictionResultCam !== "") {
       //     result = predictionResultCam;
       // } else if (predictionResultForm && predictionResultForm !== "") {
       //     result = predictionResultForm;
       // } 

       if(result == "Not a Balcony or Indoor Space") { 
            alert("Please Input a image or a balcony or indoor space");
            return;
       }


       result = result.toLowerCase().includes("balcony") ? "Balcony" : result.toLowerCase().includes("indoor") ? "Indoor" : result;

        if (!result) {
            alert("Prediction not found. Please try again.");
            return;
        }
        
        fetch("/space_identification/store_result", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ result: result })
        })
        .then(response => {
            console.log("Response status:", response.status); 
            return response.json();
        })
        .then(data => {
            if (data.success) {
                window.location.href = "/plant_prediction";  
            } else {
                alert("Failed to store prediction.");
            }
        })


    }else{
        alert("Please input an image to continue")
    }
}
let mediaStream = null;

function startWebcam() {
    const videoFeed = document.getElementById('video-feed');
    const frameIndex = document.getElementById("glassesSelector").value;
    videoFeed.src = `/video_feed/?glasses_index=${frameIndex}&t=${new Date().getTime()}`;

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            mediaStream = stream;
        })
        .catch(error => console.error("Webcam error:", error));
}

function stopWebcam() {
    const videoFeed = document.getElementById('video-feed');
    videoFeed.src = "";
    
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
}

function updateProcessedImage() {
    console.log("updateProcessedImage called");
    const frameIndex = document.getElementById("glassesSelector").value;
    console.log("Selected frame index:", frameIndex);
    
    const videoFeed = document.getElementById('video-feed');
    const processedImage = document.getElementById('processed-image');

    const formData = new FormData();
    formData.append('glasses_index', frameIndex);
    formData.append('csrfmiddlewaretoken', document.querySelector('[name=csrfmiddlewaretoken]').value);

    // Update video feed if it's active
    if (videoFeed && videoFeed.src.includes('video_feed')) {
        console.log("Updating video feed");
        videoFeed.src = `/video_feed/?glasses_index=${frameIndex}&t=${new Date().getTime()}`;
    }

    // Update processed image if it exists
    if (processedImage) {
        console.log("Updating processed image");
        fetch("/upload/", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log("Received response:", data);
            if (data.processed_url) {
                processedImage.src = data.processed_url + "?t=" + new Date().getTime();
            } else {
                console.error("Error:", data.error);
            }
        })
        .catch(error => console.error("Fetch error:", error));
    }
} 
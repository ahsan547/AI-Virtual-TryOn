const video = document.getElementById("webcam")
const userImage = document.getElementById("user-image")
const overlay = document.getElementById("overlay")
const ctx = overlay.getContext("2d")

let faceMesh
let selectedGlasses

async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true })
  video.srcObject = stream
  await video.play()
  overlay.width = video.videoWidth
  overlay.height = video.videoHeight
}

async function loadFaceMesh() {
  const { FaceMesh } = await import("@mediapipe/face_mesh")
  faceMesh = new FaceMesh({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    },
  })
  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  })
  faceMesh.onResults(onResults)
}

function onResults(results) {
  ctx.clearRect(0, 0, overlay.width, overlay.height)
  if (results.multiFaceLandmarks) {
    for (const landmarks of results.multiFaceLandmarks) {
      drawGlasses(landmarks)
    }
  }
}

function drawGlasses(landmarks) {
  if (!selectedGlasses) return

  const nose = landmarks[5]
  const leftEye = landmarks[33]
  const rightEye = landmarks[263]

  const glassesWidth = Math.abs(leftEye.x - rightEye.x) * overlay.width * 1.5
  const glassesHeight = glassesWidth * 0.3

  const x = leftEye.x * overlay.width - glassesWidth * 0.25
  const y = nose.y * overlay.height - glassesHeight * 0.5

  ctx.drawImage(selectedGlasses, x, y, glassesWidth, glassesHeight)
}

function selectGlasses(glassesUrl) {
  selectedGlasses = new Image()
  selectedGlasses.src = glassesUrl
  selectedGlasses.onload = () => {
    if (video.srcObject) {
      faceMesh.send({ image: video })
    } else {
      faceMesh.send({ image: userImage })
    }
  }
}

async function init() {
  await loadFaceMesh()
  if (video) {
    await setupWebcam()
    faceMesh.send({ image: video })
  } else {
    faceMesh.send({ image: userImage })
  }
}

init()

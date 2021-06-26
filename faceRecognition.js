const { canvas } = require("./canvas.js");
const { faceDetectionOptions } = require("./faceDetection.js");
const { faceDetectionNet } = require("./faceDetection.js");
const express = require("express");
const tf = require('@tensorflow/tfjs-node')
const faceapi = require('@vladmandic/face-api');

const router = express.Router();

async function run(avatar, REFERENCE_IMAGE, profileId) {
  console.log('in function run', avatar, REFERENCE_IMAGE)
  await faceDetectionNet.loadFromDisk("./models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
  await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");

    let QUERY_IMAGE = avatar;
    const referenceImage = await canvas.loadImage(REFERENCE_IMAGE); 
    const queryImage = await canvas.loadImage(QUERY_IMAGE);
   console.log({queryImage, referenceImage})
    // detect faces
    const resultsRef = await faceapi
      .detectAllFaces(referenceImage, faceDetectionOptions)
      .withFaceLandmarks()
      .withFaceDescriptors();
    console.log({resultsRef})
    const resultsQuery = await faceapi
      .detectAllFaces(queryImage, faceDetectionOptions)
      .withFaceLandmarks()
      .withFaceDescriptors();
    console.log({resultsQuery})
    const faceMatcher = new faceapi.FaceMatcher(resultsRef, 0.55);
    console.log({faceMatcher})

    const labels = faceMatcher.labeledDescriptors.map((ld) => ld.label);

    resultsQuery.map((res) => {
      const bestMatch = faceMatcher.findBestMatch(res.descriptor);
      if (bestMatch._label === "person 1") {
        console.log('its a match')
      } else {
        console.log('no match')
      }
    });
  return ;
}
//run("https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/Thor/2.jpg", "https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/Hawkeye/1.jpg")
router.post("/face", async (req, res) => {
  
  const obj = req.body;
  console.log(obj)
  const REFERENCE_IMAGE = obj.objImages.selfieUrl;
  const avatar = obj.objImages.avatar
  const profileId = obj.profileId
  console.log({avatar, REFERENCE_IMAGE, profileId})
  try {
    const newArray = await run(avatar, REFERENCE_IMAGE, profileId);
    if (newArray) {
      res.status(200).json(newArray);
    }
  } catch (err) {
    res.status(500).json(err);
  }
});

router.get("/", async (req, res) => {
  res.status(200).json('Welcome to face recognition.')
});
module.exports = router;
